import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse
import cv2
import os
import random

# --- 1. 配置參數 ---
IMAGE_SIZE = (224, 224)  # 確保與訓練時的尺寸一致

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (e.g., 'dataset/test/' or 'dataset/validation/')")
ap.add_argument("-m", "--model", required=True,
                help="path to trained model file (e.g., 'animal_classifier.h5')")
ap.add_argument("-n", "--num-images", type=int, default=10,
                help="number of random images to predict")
args = vars(ap.parse_args())

# initialize the class labels
class_labels = ["cat", "dog", "panda"]  # 確保與訓練時的類別順序一致

# --- 2. 載入模型 ---
print("[INFO] 載入預訓練網路中...")
model = load_model(args["model"])

# --- 3. 數據預處理 (用於預測) ---
# 單張圖片的預處理，只需要手動歸一化即可，不需要ImageDataGenerator對象
# predict_datagen = ImageDataGenerator(rescale=1./255) # 這一行其實不需要，但保留也不會錯

# --- 4. 隨機選擇圖片並預測 ---
print("[INFO] 載入圖片中...")
# 獲取指定數據集目錄下的所有圖片路徑
image_paths = []
for root, _, files in os.walk(args["dataset"]):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_paths.append(os.path.join(root, file))

# 隨機抽取指定數量的圖片
if len(image_paths) == 0:
    print(f"錯誤：在目錄 '{args['dataset']}' 中未找到任何圖片。")
    exit()

random_image_paths = random.sample(image_paths, min(args["num_images"], len(image_paths)))
print(f"將預測 {len(random_image_paths)} 張隨機圖片...")

# 遍歷每張選取的圖片
for i, image_path in enumerate(random_image_paths):
    # 載入原始圖片用於顯示
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"無法載入圖片: {image_path}")
        continue

    # 載入並預處理圖片以供模型預測
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # 手動歸一化像素值到 [0, 1] 範圍

    # 進行預測
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    predicted_label = class_labels[predicted_class_index]
    label_text = f"{predicted_label}: {confidence * 100:.2f}%"

    # 在原始圖片上繪製標籤
    # 將原始圖片縮放以適應顯示，並保留長寬比
    display_width = 400
    display_height = int(original_image.shape[0] * (display_width / original_image.shape[1]))
    display_image = cv2.resize(original_image, (display_width, display_height))

    # 計算文字位置
    font_scale = 0.8
    font_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                          font_thickness)

    # 繪製半透明背景方塊
    overlay = display_image.copy()
    cv2.rectangle(overlay, (0, 0), (text_width + 20, text_height + baseline + 20), (0, 0, 0), -1)
    alpha = 0.6
    display_image = cv2.addWeighted(overlay, alpha, display_image, 1 - alpha, 0)

    cv2.putText(display_image, label_text, (10, text_height + 10),  # (x, y) coordinates
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)  # Green color, thickness

    # 顯示圖片
    cv2.imshow(f"Prediction {i + 1}", display_image)
    cv2.waitKey(0)  # Wait for a key press to close the window

cv2.destroyAllWindows()  # Close all OpenCV windows after all images are processed