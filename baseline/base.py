import numpy as np
import pandas as pd
import os
import tensorflow as tf

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

    initial_learning_rate = 0.001

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps=5000, alpha=0.0001
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss='mse',  # Assuming you want the same custom loss
        metrics=['mse']
    )

    return model





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

checkpoint_path = "checkpoint/vilbert.h5"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

repeats = 5
results = []
for i in range(repeats):
    model = build_model((X_train.shape[1]))

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.3, min_lr=1e-6, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, verbose=1, restore_best_weights=True)


    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=256,
        epochs=4000,
        callbacks=[reduce_lr, checkpoint, early_stopping],
        verbose=1
    )
    model = tf.keras.models.load_model(checkpoint_path)

    y_pred = model.predict(X_test).flatten()

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test.flatten(), y_pred.flatten())
    spearman_corr, _ = spearmanr(y_test.flatten(), y_pred.flatten())
    kendall_corr_c, _ = kendalltau(y_test.flatten(), y_pred.flatten(), variant='c')

    # 输出结果
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")
    print(f"Kendall's Tau c: {kendall_corr_c:.4f}")

    results.append({"mse": mse, "mae": mae, "pearson": pearson_corr, "spearman": spearman_corr, "kendall": kendall_corr_c})
pd.DataFrame(results).to_csv("../results/baseline.csv")

