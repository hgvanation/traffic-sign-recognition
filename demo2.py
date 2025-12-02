import warnings 
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# Cấu hình 
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50

train_imgs = r"D:\New folder (3)\Train_processed-20251202T065008Z-1-001\Train_processed"
test_imgs  = r"D:\New folder (3)\Test\Test"

# Đọc dữ liệu 
train_df = pd.read_csv(r"D:\New folder (3)\Train.csv")
test_df  = pd.read_csv(r"D:\New folder (3)\Test.csv")

# Chuẩn hóa lại đường dẫn ảnh
train_df["Path"] = train_df["Path"].str.replace("Train/", "", regex=False)
test_df["Path"]  = test_df["Path"].str.replace("Test/", "", regex=False)

# Chuẩn bị nhãn
train_df["ClassId_str"] = train_df["ClassId"].astype(str)
test_df["ClassId_str"]  = test_df["ClassId"].astype(str)

# Chia tập train/validation
train_data, valid_data = train_test_split(
    train_df, test_size=0.2, stratify=train_df["ClassId"], random_state=42
)

# Tạo bảng lookup nhãn
unique_labels = sorted(train_data["ClassId_str"].unique())
num_classes = len(unique_labels)

keys_tensor = tf.constant(unique_labels)
vals_tensor = tf.constant(list(range(num_classes)))
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
    default_value=-1
)


#  Hàm load và xử lý ảnh 
def load_and_preprocess_img(path, label, base_dir):
    full_path = tf.strings.join([base_dir, "/", path])
    img = tf.io.read_file(full_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    label_id = table.lookup(label)
    label_onehot = tf.one_hot(label_id, num_classes)
    return img, label_onehot

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# Hàm tạo dataset
def make_dataset(df, base_dir, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((df["Path"].values, df["ClassId_str"].values))
    ds = ds.map(lambda x, y: load_and_preprocess_img(x, y, base_dir),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Tạo dataset
train_gen_aug = make_dataset(train_data, train_imgs, augment=True, shuffle=True)
valid_gen_aug = make_dataset(valid_data, train_imgs)
test_gen_aug  = make_dataset(test_df, test_imgs)

print(f" Dataset đã sẵn sàng với {num_classes} lớp")

# Tính class weights 
class_weight = compute_class_weight('balanced', classes=np.unique(train_df['ClassId']),
                                   y=train_df['ClassId'])
class_weights_dict = dict(enumerate(class_weight))

# Model CNN cũ 
model_aug = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), padding="same"),
    layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
    layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
model_aug.summary()

model_aug.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback 
best_model_path = r"D:\New folder (3)\best_model1_1_checkpoint.keras"

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, min_lr=1e-5, verbose=1)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

print(f" Model tốt nhất sẽ được lưu tại: {best_model_path}")

# Train 
history_aug = model_aug.fit(
    train_gen_aug,
    validation_data=valid_gen_aug,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr, checkpoint_callback],
    class_weight=class_weights_dict
)



# Đánh giá 
test_loss, test_accuracy = model_aug.evaluate(test_gen_aug)
print(f" Test accuracy: {test_accuracy*100:.2f}%")
print(f" Test loss: {test_loss:.3f}")

valid_loss, valid_accuracy = model_aug.evaluate(valid_gen_aug)
print(f" Validation accuracy: {valid_accuracy*100:.2f}%")
print(f" Validation loss: {valid_loss:.3f}")

# Dự đoán & Báo cáo
pred_probs = model_aug.predict(test_gen_aug)
pred_labels = np.argmax(pred_probs, axis=1)

print("\n Classification report:")
print(classification_report(np.argmax(np.array(list(test_gen_aug.map(lambda x, y: y))).sum(axis=0), axis=1),
                            pred_labels))
