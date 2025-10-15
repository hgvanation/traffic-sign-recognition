import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout,BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from skimage.measure import shannon_entropy

data = np.load("D:\\thị giác\\dataset.npz")

train_images = data["train_images"]
train_labels = data["train_labels"]
test_images  = data["test_images"]
test_labels  = data["test_labels"]

def enhance_image(img):
    # 1️⃣ Chuyển sang YUV và cân bằng histogram nhẹ hơn
    img_yuv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    y = img_yuv[:, :, 0]
    # Cân bằng histogram có trọng số, không để quá sáng
    y_eq = cv2.equalizeHist(y)
    img_yuv[:, :, 0] = cv2.addWeighted(y, 0.7, y_eq, 0.3, 0)  # 70% gốc + 30% cân bằng
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # 2️⃣ Sharpen nhẹ
    gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    sharpened = cv2.addWeighted(img, 1.1, gaussian, -0.1, 0)  # Giảm cường độ sharpen

    # 3️⃣ Tăng cường biên vừa phải
    gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # 4️⃣ Trộn nhẹ với ảnh gốc
    enhanced = cv2.addWeighted(sharpened, 0.97, edges_colored, 0.03, 0)

    # 5️⃣ Chuẩn hóa [0, 1]
    enhanced = enhanced.astype("float32") / 255.0
    return enhanced

# Áp dụng cho toàn bộ tập train
train_images_enhanced = np.zeros_like(train_images)
for i in tqdm(range(len(train_images))):
    train_images_enhanced[i] = enhance_image(train_images[i])
    
idx = np.random.randint(0, len(train_images))
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(train_images[idx])
plt.title("Ảnh gốc sau tiền xử lý")

plt.subplot(1,2,2)
plt.imshow(train_images_enhanced[idx])
plt.title("Ảnh sau tăng cường (enhanced)")
#plt.show()

def image_quality_metrics(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = np.std(gray)
    entropy = shannon_entropy(gray)
    return sharpness, contrast, entropy

# So sánh 1 ảnh gốc và 1 ảnh enhanced
sharp_orig, cont_orig, ent_orig = image_quality_metrics(train_images[0])
sharp_enh, cont_enh, ent_enh = image_quality_metrics(train_images_enhanced[0])

print("GỐC:")
print(f"  Sharpness: {sharp_orig:.2f}, Contrast: {cont_orig:.2f}, Entropy: {ent_orig:.2f}")
print("TĂNG CƯỜNG:")
print(f"  Sharpness: {sharp_enh:.2f}, Contrast: {cont_enh:.2f}, Entropy: {ent_enh:.2f}")

sharp_orig_list, cont_orig_list, ent_orig_list = [], [], []
sharp_enh_list, cont_enh_list, ent_enh_list = [], [], []

for i in range(39209):  # ví dụ tính trên 100 ảnh đầu tiên
    s, c, e = image_quality_metrics(train_images[i])
    sharp_orig_list.append(s)
    cont_orig_list.append(c)
    ent_orig_list.append(e)

    s, c, e = image_quality_metrics(train_images_enhanced[i])
    sharp_enh_list.append(s)
    cont_enh_list.append(c)
    ent_enh_list.append(e)

print("Trung bình GỐC:", np.mean(sharp_orig_list), np.mean(cont_orig_list), np.mean(ent_orig_list))
print("Trung bình TĂNG CƯỜNG:", np.mean(sharp_enh_list), np.mean(cont_enh_list), np.mean(ent_enh_list))

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

# Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=train_images.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.15))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.20))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(43, activation='softmax'))

# Compilation of the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model display
model.summary()

# === 🔹 Thêm phần ModelCheckpoint ngay trước khi train ===
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath=r"D:\thị giác\best_model_traffic_sign.keras",   # 🔹 nơi lưu model tốt nhất
    monitor='val_accuracy',                     # 🔹 theo dõi độ chính xác validation
    save_best_only=True,                        # 🔹 chỉ lưu khi tốt hơn model trước
    mode='max',
    verbose=1
)

# === Training the Model ===
with tf.device('/GPU:0'):
    epochs = 35
    history1 = model.fit(
        train_images, train_labels,
        batch_size=128,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        callbacks=[checkpoint]   # 🔹 thêm callback vào đây
    )


def plot_accuracy_loss(history):
    """
    Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(12,5))

    # Plot accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], 'bo--', label='train_accuracy')
    plt.plot(history.history['val_accuracy'], 'ro--', label='val_accuracy')
    plt.title("Train vs Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], 'bo--', label='train_loss')
    plt.plot(history.history['val_loss'], 'ro--', label='val_loss')
    plt.title("Train vs Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


plot_accuracy_loss(history1)

pred_probs = model.predict(test_images)         
pred_labels = np.argmax(pred_probs, axis=1)