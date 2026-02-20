import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.regularizers import l2
from scipy.stats import pearsonr, spearmanr, kendalltau
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


def create_pairwise_data(X, y, max_pairs=2000):
    """Generate a fixed number of pairwise training pairs efficiently."""
    num_features = X.shape[1]

    # Pre-allocate memory
    pairwise_X1 = np.empty((max_pairs, num_features), dtype=np.float32)
    pairwise_X2 = np.empty((max_pairs, num_features), dtype=np.float32)
    pairwise_Y = np.empty((max_pairs,), dtype=np.int8)

    indices = np.arange(len(y))
    pair_count = 0
    seen = set()
    while pair_count < max_pairs:
        i, j = np.random.choice(indices, 2, replace=False)
        if (i,j) not in seen:
            if y[i] != y[j]:  # Ensure different ratings
                pairwise_X1[pair_count] = X[i]
                pairwise_X2[pair_count] = X[j]
                pairwise_Y[pair_count] = 1 if y[i] > y[j] else -1
                pair_count += 1
                seen.add((i,j))
                seen.add((j,i))

    return pairwise_X1, pairwise_X2, pairwise_Y

# Load dataset
file_path = r"../data/VICR_entire.csv"
df = pd.read_csv(file_path)

# Convert text embedding column to numpy arrays
df["Concatnated_image_caption"] = df["Concatnated_image_caption"].apply(lambda x: np.array(eval(x)))
df["Rating"] = df["Rating"].apply(lambda x: float(x))
# Normalize ratings
# df["Rating"] = (df["Rating"] - df["Rating"].min()) / (df["Rating"].max() - df["Rating"].min())

# Extract features and target variable
X = np.vstack(df["Concatnated_image_caption"].values)
y = df["Rating"].values.reshape(-1, 1)

num_samples = len(X)
split_index = int(0.8 * num_samples)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 归一化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

N = 1

# Generate pairwise training set
X1, X2, Y = create_pairwise_data(X_train, y_train, N*len(y_train))

# Initialize encoder
encoder = create_encoder(input_dim=X1.shape[1])

# Build dual encoder model
pairwise_model = DualEncoderText(encoder)
pairwise_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Prepare dataset as a dictionary for the model
train_data = {"A": X1, "B": X2, "Label": Y}

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop if validation loss doesn't improve for 5 epochs
    restore_best_weights=True,
    verbose=1
)

# Train model
history = pairwise_model.fit(train_data, epochs=30, batch_size=32, validation_split=0.2,
                             callbacks=[early_stopping])

# Compute utility scores for all test samples
# predicted_scores = np.array(
#     [pairwise_model.encoder(np.expand_dims(xi, axis=0), training=False).numpy().flatten()[0] for xi in X_test])
y_pred = pairwise_model.encoder.predict(X_test).flatten()

# Compute Pearson and Spearman correlation
pearson_corr, _ = pearsonr(y_test.flatten(), y_pred.flatten())
spearman_corr, _ = spearmanr(y_test.flatten(), y_pred.flatten())
kendall_corr, _ = kendalltau(y_test.flatten(), y_pred.flatten())
kendall_corr_c, _ = kendalltau(y_test.flatten(), y_pred.flatten(), variant='c')

# Print the results
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")
print(f"Kendall's Tau: {kendall_corr:.4f}")
print(f"Kendall's Tau c: {kendall_corr_c:.4f}")

