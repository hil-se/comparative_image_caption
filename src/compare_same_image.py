import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr, kendalltau
from embeddings_serialize import Image_Caption_Embedding, serialize, deserialize


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
    


class DualEncoderText(tf.keras.Model):
    def __init__(self, encoder, margin=1.5):
        super().__init__()
        self.encoder = encoder
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):
        A = self.encoder(features["A"], training=training)
        B = self.encoder(features["B"], training=training)
        return A, B, features["Label"]

    def compute_loss(self, A, B, y):
        pred = tf.squeeze(A - B)
        y = tf.cast(y, tf.float32)
        return tf.reduce_mean(tf.maximum(0.0, self.margin - y * pred))

    def train_step(self, features):
        with tf.GradientTape() as tape:
            A, B, y = self(features, training=True)
            loss = self.compute_loss(A, B, y)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        A, B, y = self(features, training=False)
        loss = self.compute_loss(A, B, y)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def predict_pairs(self, A, B):
        return self.encoder(A, training=False) - self.encoder(B, training=False)



# Load ViLBERT embeddings


with open("embeddings/VICR-train-vilbert.emb", 'rb') as f:
    train_raw = deserialize(f)

with open("embeddings/VICR-val-vilbert.emb", 'rb') as f:
    val_raw = deserialize(f)

with open("embeddings/VICR-test-vilbert.emb", 'rb') as f:
    test_raw = deserialize(f)

all_data = train_raw + val_raw + test_raw




all_embeddings = []
all_ratings = []
all_images = []

for rs in all_data:
    rating = np.mean(rs.ratings)
    embedding = np.concatenate([rs.image_embedding, rs.caption_embedding])
    all_embeddings.append(embedding)
    all_ratings.append(rating)
    all_images.append(rs.image)

all_embeddings = np.array(all_embeddings, dtype=np.float32)
all_ratings = np.array(all_ratings, dtype=np.float32)
all_images = np.array(all_images)

all_ratings = (all_ratings - all_ratings.min()) / (all_ratings.max() - all_ratings.min())


scaler = StandardScaler()
all_embeddings_scaled = scaler.fit_transform(all_embeddings)


# generate same image pairs

image_to_indices = defaultdict(list)
for idx, image in enumerate(all_images):
    image_to_indices[image].append(idx)

X1_raw, X2_raw, Y_pairs = [], [], []

for image, indices in image_to_indices.items():
    if len(indices) < 2:
        continue

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_i, idx_j = indices[i], indices[j]
            r1, r2 = all_ratings[idx_i], all_ratings[idx_j]

            if r1 == r2 or abs(r1 - r2) < 0.05:
                continue

            e1 = all_embeddings_scaled[idx_i]
            e2 = all_embeddings_scaled[idx_j]
            label = 1 if r1 > r2 else -1

            X1_raw.append(e1); X2_raw.append(e2); Y_pairs.append(label)
            X1_raw.append(e2); X2_raw.append(e1); Y_pairs.append(-label)

X1_all = np.array(X1_raw, dtype=np.float32)
X2_all = np.array(X2_raw, dtype=np.float32)
Y_all  = np.array(Y_pairs, dtype=np.int8)




X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(
    X1_all, X2_all, Y_all, test_size=0.2, random_state=42
)

X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(
    all_embeddings_scaled, all_ratings, test_size=0.2, random_state=42
)


# Training and evaluation

results = []

for run in range(1, 6):
    print(f"\n========== Run {run} ==========")

    encoder = create_encoder(input_dim=X1_all.shape[1])
    model = DualEncoderText(encoder, margin=1.5)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    train_data = {"A": X1_train, "B": X2_train, "Label": Y_train}

    model.fit(
        train_data,
        epochs=25,
        batch_size=256,
        verbose=1
    )



    pred_diff = model.predict_pairs(X1_test, X2_test).numpy().flatten()
    pred_labels = np.where(pred_diff > 0, 1, -1)
    p_o = np.mean(pred_labels == Y_test)

    pred_scores_flat = encoder.predict(X_test_flat, verbose=0).flatten()

    pearson_corr, _ = pearsonr(pred_scores_flat, y_test_flat)
    spearman_corr, _ = spearmanr(pred_scores_flat, y_test_flat)
    kendall_corr_c, _ = kendalltau(pred_scores_flat, y_test_flat, variant='c')

    print(f"Observed Agreement (Po): {p_o:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")
    print(f"Kendall's Tau c: {kendall_corr_c:.4f}")

    results.append({"run": run, "p_o": p_o, "pearson": pearson_corr, "spearman": spearman_corr,"kendall_c": kendall_corr_c})

df_results = pd.DataFrame(results)

df_results.to_csv("../results/compare_same_image.csv", index=False)

print("\n========== SUMMARY ==========")
print(df_results)

