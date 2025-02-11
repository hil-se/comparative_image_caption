import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import os
import matplotlib.pyplot as plt

# Load dataset
file_path = r"..\data\VICR_Sample250_Cleaned.csv"
df = pd.read_csv(file_path)

df["Concatnated_image_caption"] = df["Concatnated_image_caption"].apply(lambda x: np.array(eval(x)))

# Normalize ratings between 0-1
df["Rating"] = (df["Rating"] - df["Rating"].min()) / (df["Rating"].max() - df["Rating"].min())

# Extract features (X) and target variable (y)
X = np.vstack(df["Concatnated_image_caption"].values)
y = df["Rating"].values.reshape(-1, 1)

def generate_ranking_features(X):
    mean_features = np.mean(X, axis=1, keepdims=True)
    std_features = np.std(X, axis=1, keepdims=True)
    min_features = np.min(X, axis=1, keepdims=True)
    max_features = np.max(X, axis=1, keepdims=True)
    return np.hstack([X, mean_features, std_features, min_features, max_features])

X = generate_ranking_features(X)


scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Label Smoothing (prevent overfitting)
y_train = y_train * 0.9 + 0.05

# Custom Ranking Penalized MAE Loss
def ranking_penalized_mae(y_true, y_pred):
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)

    # Rank Penalty: Penalizes incorrect ranking relationships
    y_true_diff = tf.expand_dims(y_true, axis=1) - tf.expand_dims(y_true, axis=0)
    y_pred_diff = tf.expand_dims(y_pred, axis=1) - tf.expand_dims(y_pred, axis=0)
    rank_penalty = tf.reduce_mean(tf.square(tf.sign(y_true_diff) - tf.sign(y_pred_diff)))

    return mae + 0.3 * rank_penalty  # Increased weight of ranking penalty

# Define Model with Optimized Architecture
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2048, kernel_regularizer=l2(1e-5), input_shape=input_shape),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1024, kernel_regularizer=l2(1e-5)),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(512, kernel_regularizer=l2(1e-5)),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(256, kernel_regularizer=l2(1e-5)),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(128, kernel_regularizer=l2(1e-5)),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1, activation=None)  
    ])

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps=5000, alpha=0.0001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=ranking_penalized_mae,
        metrics=['mae']
    )

    return model

# Build and compile model
model = build_model((X_train.shape[1],))

# Ensure checkpoint directory exists
checkpoint_path = "checkpoint/image_caption.keras"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# Define callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.3, min_lr=1e-6, verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=500,
    callbacks=[reduce_lr, checkpoint, early_stopping],
    verbose=1
)

# Load best model after training
print("\nLoading best checkpoint model...")
model = tf.keras.models.load_model(checkpoint_path, custom_objects={"ranking_penalized_mae": ranking_penalized_mae})

# Make predictions
y_pred = model.predict(X_test).flatten()

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
pearson_corr, _ = pearsonr(y_test.flatten(), y_pred.flatten())
spearman_corr, _ = spearmanr(y_test.flatten(), y_pred.flatten())

# results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")

# Plot Training & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()
