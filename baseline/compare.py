import numpy as np
import os
from pdb import set_trace
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, kendalltau

from embeddings_serialize import Image_Caption_Embedding, serialize, deserialize


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),

        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),

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




X_train = []
y_train = []

with open("embeddings/VICR-train-vilbert.emb", 'rb') as in_file:
    result = deserialize(in_file)
for rs in result:
    rating = np.mean(rs.ratings).round()
    embedding = np.concatenate([rs.image_embedding, rs.caption_embedding]).tolist()
    y_train.append(rating)
    X_train.append(embedding)

with open("embeddings/VICR-val-vilbert.emb", 'rb') as in_file:
    result = deserialize(in_file)
for rs in result:
    rating = np.mean(rs.ratings).round()
    embedding = np.concatenate([rs.image_embedding, rs.caption_embedding]).tolist()
    y_train.append(rating)
    X_train.append(embedding)

X_test = []
y_test = []

with open("embeddings/VICR-test-vilbert.emb", 'rb') as in_file:
    result = deserialize(in_file)
for rs in result:
    rating = np.mean(rs.ratings).round()
    embedding = np.concatenate([rs.image_embedding, rs.caption_embedding])
    y_test.append(rating)
    X_test.append(embedding)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

checkpoint_path = "checkpoint/compare.keras"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

results = []
for N in range(1,6):
    for i in range(5):
        # Generate pairwise training set
        X1, X2, Y = create_pairwise_data(X_train, y_train, N*len(y_train))

        # Initialize encoder
        encoder = build_model(input_dim=X1.shape[1])

        # Build dual encoder model
        pairwise_model = DualEncoderText(encoder)
        initial_learning_rate = 0.001

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate, decay_steps=5000, alpha=0.0001
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        pairwise_model.compile(optimizer=optimizer)

        # Prepare dataset as a dictionary for the model
        train_data = {"A": X1, "B": X2, "Label": Y}



        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.3, min_lr=1e-6, verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, verbose=1, restore_best_weights=True)

        # Train model
        history = pairwise_model.fit(train_data, epochs=500, batch_size=256, validation_split=0.2, callbacks=[reduce_lr, checkpoint, early_stopping])


        y_pred = pairwise_model.encoder.predict(X_test).flatten()

        # Compute Pearson and Spearman correlation
        pearson_corr, _ = pearsonr(y_test.flatten(), y_pred.flatten())
        spearman_corr, _ = spearmanr(y_test.flatten(), y_pred.flatten())
        kendall_corr, _ = kendalltau(y_test.flatten(), y_pred.flatten())
        kendall_corr_c, _ = kendalltau(y_test.flatten(), y_pred.flatten(), variant='c')


        # 输出结果
        print(f"Pearson Correlation: {pearson_corr:.4f}")
        print(f"Spearman Correlation: {spearman_corr:.4f}")
        print(f"Kendall's Tau c: {kendall_corr_c:.4f}")

        results.append({"N": N, "pearson": pearson_corr, "spearman": spearman_corr, "kendall": kendall_corr_c})
pd.DataFrame(results).to_csv("../results/compare.csv")