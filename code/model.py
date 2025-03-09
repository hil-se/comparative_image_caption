import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.regularizers import l2
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

# 设置随机种子，确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

def create_encoder(input_dim):
    """Creates a feedforward encoder model for pairwise ranking."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),

        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-5)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1, activation=None)
    ])
    return model


class DualEncoderText(tf.keras.Model):
    def __init__(self, encoder, **kwargs):
        super(DualEncoderText, self).__init__(**kwargs)
        self.encoder = encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, trainable=True):
        encodings_A = self.encoder(features["A"], training=trainable)
        encodings_B = self.encoder(features["B"], training=trainable)
        y = features["Label"]
        return encodings_A, encodings_B, y

    def compute_loss(self, encodings_A, encodings_B, y):
        encodings_A = tf.squeeze(encodings_A)
        encodings_B = tf.squeeze(encodings_B)
        pred = encodings_A - encodings_B
        y = tf.cast(y, tf.float32)

        # Hinge loss: max(0, 1 - y * (f(xi) - f(xj)))
        loss = tf.reduce_mean(tf.math.maximum(0.0, 1.0 - (y * pred)))

        return loss

    def train_step(self, feature):
        with tf.GradientTape() as tape:
            encodings_A, encodings_B, y = self(feature, trainable=True)
            loss = self.compute_loss(encodings_A, encodings_B, y)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, feature):
        encodings_A, encodings_B, y = self(feature, trainable=False)
        loss = self.compute_loss(encodings_A, encodings_B, y)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def predict(self, A, B):
        """Predicts preference between two items."""
        return self.encoder(A) - self.encoder(B)


# Load dataset
file_path = r"C:\Users\29049\Desktop\新建文件夹\Final_VICR.csv"
df = pd.read_csv(file_path)

# Convert text embedding column to numpy arrays
df["Concatnated_image_caption"] = df["Concatnated_image_caption"].apply(lambda x: np.array(eval(x)))

# Normalize ratings
df["Rating"] = (df["Rating"] - df["Rating"].min()) / (df["Rating"].max() - df["Rating"].min())

# Extract features and target variable
X = np.vstack(df["Concatnated_image_caption"].values)
y = df["Rating"].values.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)


def create_pairwise_data(X, y, max_pairs=43560000):
    """Generate a fixed number of pairwise training pairs efficiently."""
    num_features = X.shape[1]

    # Pre-allocate memory
    pairwise_X1 = np.empty((max_pairs, num_features), dtype=np.float32)
    pairwise_X2 = np.empty((max_pairs, num_features), dtype=np.float32)
    pairwise_Y = np.empty((max_pairs,), dtype=np.int8)

    indices = np.arange(len(y))
    pair_count = 0

    while pair_count < max_pairs:
        i, j = np.random.choice(indices, 2, replace=False)
        if y[i] != y[j]:  # Ensure different ratings
            pairwise_X1[pair_count] = X[i]
            pairwise_X2[pair_count] = X[j]
            pairwise_Y[pair_count] = 1 if y[i] > y[j] else -1
            pair_count += 1

    return pairwise_X1, pairwise_X2, pairwise_Y


# Generate pairwise dataset
X1, X2, Y = create_pairwise_data(X, y)

# 固定按顺序划分训练集和测试集（前 80% 训练，后 20% 测试）
num_samples = len(X1)
split_index = int(0.8 * num_samples)

X1_train, X1_test = X1[:split_index], X1[split_index:]
X2_train, X2_test = X2[:split_index], X2[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# Initialize encoder
encoder = create_encoder(input_dim=X1_train.shape[1])

# Build dual encoder model
pairwise_model = DualEncoderText(encoder)
pairwise_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Prepare dataset as a dictionary for the model
train_data = {"A": X1_train, "B": X2_train, "Label": Y_train}
test_data = {"A": X1_test, "B": X2_test, "Label": Y_test}

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop if validation loss doesn't improve for 5 epochs
    restore_best_weights=True,
    verbose=1
)

# Train model
history = pairwise_model.fit(train_data, epochs=30, batch_size=32, validation_data=test_data,
                             callbacks=[early_stopping])

# Evaluate performance
test_loss = pairwise_model.evaluate(test_data)
print(f"Test Loss: {test_loss:.4f}")

# Predict preference between two samples
sample_A, sample_B = X1_test[0], X2_test[0]
preference = pairwise_model.predict(np.expand_dims(sample_A, axis=0), np.expand_dims(sample_B, axis=0))
print(f"Preference score (A - B): {preference}")

# Compute utility scores for all test samples
predicted_scores = np.array(
    [pairwise_model.encoder(np.expand_dims(xi, axis=0), training=False).numpy().flatten()[0] for xi in X1_test])

# Compute Pearson and Spearman correlation
pearson_corr, _ = pearsonr(Y_test, predicted_scores)
spearman_corr, _ = spearmanr(Y_test, predicted_scores)

# Print the results
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")

