import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# === 固定随机种子确保可复现 ===
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# === 数据加载与预处理 ===
file_path = r"C:\Users\29049\Desktop\新建文件夹\VICR_image_duplicates_diff_rating.csv"
df = pd.read_csv(file_path)
df["Concatnated_image_caption"] = df["Concatnated_image_caption"].apply(lambda x: np.array(eval(x)))

# 归一化评分
df["Rating"] = (df["Rating"] - df["Rating"].min()) / (df["Rating"].max() - df["Rating"].min())

X = np.vstack(df["Concatnated_image_caption"].values)
y = df["Rating"].values.reshape(-1, 1)

# 提取统计特征
def generate_ranking_features(X):
    mean_features = np.mean(X, axis=1, keepdims=True)
    std_features = np.std(X, axis=1, keepdims=True)
    min_features = np.min(X, axis=1, keepdims=True)
    max_features = np.max(X, axis=1, keepdims=True)
    return np.hstack([X, mean_features, std_features, min_features, max_features])

X = generate_ranking_features(X)

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练/测试集
num_samples = len(X)
split_index = int(0.8 * num_samples)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 避免极值
y_train = y_train * 0.9 + 0.05

# === 自定义排序惩罚损失 ===
def ranking_penalized_mae(y_true, y_pred):
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    y_true_diff = tf.expand_dims(y_true, axis=1) - tf.expand_dims(y_true, axis=0)
    y_pred_diff = tf.expand_dims(y_pred, axis=1) - tf.expand_dims(y_pred, axis=0)
    rank_penalty = tf.reduce_mean(tf.square(tf.sign(y_true_diff) - tf.sign(y_pred_diff)))
    return mae + 0.3 * rank_penalty

# === 构建模型 ===
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

    model.compile(optimizer=optimizer, loss=ranking_penalized_mae, metrics=['mae'])
    return model

# === 训练模型 ===
model = build_model((X_train.shape[1],))
checkpoint_path = "checkpoint/image_caption.keras"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, verbose=1, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=500,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# === 加载最佳模型进行评估 ===
print("\nLoading best checkpoint model...")
model = tf.keras.models.load_model(checkpoint_path, custom_objects={"ranking_penalized_mae": ranking_penalized_mae})
y_pred = model.predict(X_test).flatten()

# === 评估指标 ===
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
pearson_corr, _ = pearsonr(y_test.flatten(), y_pred.flatten())
spearman_corr, _ = spearmanr(y_test.flatten(), y_pred.flatten())

print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")

# === Pairwise Ranking Accuracy ===
correct_pairs = 0
total_pairs = 0
for i in range(len(y_test)):
    for j in range(i + 1, len(y_test)):
        true_diff = y_test[i] - y_test[j]
        pred_diff = y_pred[i] - y_pred[j]
        if true_diff == 0:
            continue
        if (true_diff > 0 and pred_diff > 0) or (true_diff < 0 and pred_diff < 0):
            correct_pairs += 1
        total_pairs += 1

ranking_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
print(f"Pairwise Ranking Accuracy: {ranking_accuracy:.4f}")

# === 可视化损失曲线 ===
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()
