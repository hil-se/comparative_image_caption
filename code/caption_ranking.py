import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

# Load data
df = pd.read_csv("../data/VICR_entire.csv")
df["Concatnated_image_caption"] = df["Concatnated_image_caption"].apply(lambda x: np.array(eval(x)))
df["Rating"] = (df["Rating"] - df["Rating"].min()) / (df["Rating"].max() - df["Rating"].min())

pairs = []
for _, group in df.groupby("image_embedding"):
    if len(group) < 2:
        continue
    items = group.to_dict("records")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            vec1 = items[i]["Concatnated_image_caption"]
            vec2 = items[j]["Concatnated_image_caption"]
            rating1 = items[i]["Rating"]
            rating2 = items[j]["Rating"]
            if rating1 == rating2 or abs(rating1 - rating2) < 0.05:
                continue
            label = 1 if rating1 > rating2 else -1
            pairs.append((vec1, vec2, label))
            pairs.append((vec2, vec1, -label))


X1_raw = [a for a, _, _ in pairs]
X2_raw = [b for _, b, _ in pairs]
Y = [label for _, _, label in pairs]

X_all = X1_raw + X2_raw
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

X1 = X_all_scaled[:len(X1_raw)]
X2 = X_all_scaled[len(X1_raw):]
Y = np.array(Y)

# Train/test split
X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(X1, X2, Y, test_size=0.2, random_state=42)

# Encoder
def create_encoder(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation=None)
    ])

# Pairwise model
class DualEncoderText(tf.keras.Model):
    def __init__(self, encoder, margin=1.5):
        super().__init__()
        self.encoder = encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.margin = margin

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, trainable=True):
        A = self.encoder(features["A"], training=trainable)
        B = self.encoder(features["B"], training=trainable)
        return A, B, features["Label"]

    def compute_loss(self, A, B, y):
        pred = tf.squeeze(A - B)
        y = tf.cast(y, tf.float32)
        return tf.reduce_mean(tf.maximum(0.0, self.margin - y * pred))

    def train_step(self, features):
        with tf.GradientTape() as tape:
            A, B, y = self(features)
            loss = self.compute_loss(A, B, y)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        A, B, y = self(features, trainable=False)
        loss = self.compute_loss(A, B, y)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def predict(self, A, B):
        return self.encoder(A) - self.encoder(B)

# Compile & train
encoder = create_encoder(input_dim=X1.shape[1])
model = DualEncoderText(encoder, margin=1.5)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

train_data = {"A": X1_train, "B": X2_train, "Label": Y_train}
test_data = {"A": X1_test, "B": X2_test, "Label": Y_test}

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
model.fit(train_data, epochs=60, batch_size=4, validation_data=test_data, callbacks=[early_stop], verbose=2)

# Evaluation
pred_scores = model.predict(X1_test, X2_test).numpy().flatten()
pred_labels = np.where(pred_scores > 0, 1, -1)
accuracy = np.mean(pred_labels == Y_test)
pearson_corr = pearsonr(pred_scores, Y_test)[0]
spearman_corr = spearmanr(pred_scores, Y_test)[0]

print(f"Pairwise Accuracy on Test Set: {accuracy:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")

# Non pairwise evaluation 
print("\nEvaluating encoder on 9k image caption pairs...")

X_raw_all = np.array(df["Concatnated_image_caption"].tolist())
y_raw_all = df["Rating"].values

X_scaled_all = scaler.transform(X_raw_all)

X_train_scaled, X_test_scaled, y_train_true, y_test_true = train_test_split(
    X_scaled_all, y_raw_all, test_size=0.2, random_state=42
)

# predicting scores using the encoder above
pred_scores_test = encoder.predict(X_test_scaled).flatten()

# Evaluate
pearson_test = pearsonr(pred_scores_test, y_test_true)[0]
spearman_test = spearmanr(pred_scores_test, y_test_true)[0]

print("Non-Pairwise Evaluation on Test set:")
print(f"Pearson Correlation: {pearson_test:.4f}")
print(f"Spearman Correlation: {spearman_test:.4f}")
