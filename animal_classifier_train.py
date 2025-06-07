import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import argparse

# --- 1. 配置參數 ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 3 # 貓、狗、貓熊
EPOCHS = 10

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset (e.g., 'dataset/')")
ap.add_argument("-m", "--model", required=True,
    help="path to output model file (e.g., 'animal_classifier.h5')")
args = vars(ap.parse_args())

TRAIN_DIR = os.path.join(args["dataset"], 'train')
VALIDATION_DIR = os.path.join(args["dataset"], 'validation')

# 確保數據集目錄存在
if not os.path.exists(TRAIN_DIR):
    print(f"錯誤：訓練數據集目錄 '{TRAIN_DIR}' 不存在。請準備數據集。")
    print("請將圖片分別放在以下目錄結構：")
    print(f"{args['dataset']}/")
    print("├── train/")
    print("│   ├── cat/")
    print("│   ├── dog/")
    print("│   └── panda/")
    print("└── validation/")
    print("    ├── cat/")
    print("    ├── dog/")
    print("    └── panda/")
    exit()

# --- 2. 數據預處理與增強 ---
print("[INFO] 載入圖片中...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

class_names = list(train_generator.class_indices.keys())
print(f"偵測到的類別名稱：{class_names}")

# --- 3. 構建模型 (MobileNetV2 遷移學習) ---
print("[INFO] 編譯模型中...")
base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. 訓練模型 ---
print("[INFO] 訓練網路中...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

print("\n模型訓練完成。")

# --- 5. 繪製訓練結果 ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- 6. 模型保存 ---
print("[INFO] 序列化網路中...")
model.save(args["model"])
print(f"模型已保存為 {args['model']}")

# --- 7. 模型評估 (使用驗證集) ---
print("\n評估模型在驗證集上的性能...")
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)
print(f"驗證集損失：{val_loss:.4f}")
print(f"驗證集準確率：{val_accuracy:.4f}")

# 生成混淆矩陣和分類報告
print("\n生成混淆矩陣和分類報告...")
validation_generator.reset()
# 將 np.ceil 的結果轉換為 int
Y_pred = model.predict(validation_generator, steps=int(np.ceil(validation_generator.samples / BATCH_SIZE)))
y_pred_classes = np.argmax(Y_pred, axis=1)

y_true = validation_generator.classes

conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(y_true, y_pred_classes, target_names=class_names)
print("\n分類報告：")
print(report)
