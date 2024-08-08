import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
from PIL import Image
import zipfile
import shutil

# Constants
EXTRACT_PATH = '/home/ubuntu/Desktop/extract_apk_ML'
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32

# Function to extract APK file
def extract_apk(apk_path, extract_path):
    try:
        with zipfile.ZipFile(apk_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except Exception as e:
        print(f"Error extracting {apk_path}: {e}")
        return False
    return True

# Function to convert DEX files to images
def dex_to_image(dex_file, image_path):
    try:
        with open(dex_file, 'rb') as f:
            data = f.read()
        data = bytearray(data)
        np_array = np.array(data, dtype=np.uint8)
        size = int(len(np_array) ** 0.5)
        if size * size != len(np_array):
            np_array = np_array[:size * size]
        np_array = np_array.reshape((size, size))
        img = Image.fromarray(np_array, 'L')
        img = img.resize(IMAGE_SIZE)
        img.save(image_path)
    except Exception as e:
        print(f"Error processing {dex_file}: {e}")

# Function to process all DEX files and convert to images
def process_apk_to_images(extract_path, image_dir):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.dex'):
                dex_path = os.path.join(root, file)
                image_path = os.path.join(image_dir, f"{file}.png")
                dex_to_image(dex_path, image_path)

# Function to prepare image for prediction
def prepare_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to predict an APK file
def predict_apk(apk_path, model):
    # Step 1: Extract APK file
    if not extract_apk(apk_path, EXTRACT_PATH):
        return None, None

    # Step 2: Convert DEX files to images
    image_dir = os.path.join(EXTRACT_PATH, 'images')
    process_apk_to_images(EXTRACT_PATH, image_dir)

    # Step 3: Prepare and predict
    predictions = []
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        img_array = prepare_image(image_path)
        prediction = model.predict(img_array)
        predictions.append((image_file, prediction[0][0]))

    # Step 4: Delete temporary extracted files
    shutil.rmtree(EXTRACT_PATH)

    if not predictions:
        return None, None

    # Determine the final prediction based on average probability
    avg_prediction = np.mean([p[1] for p in predictions])
    label = 'Malware' if avg_prediction > 0.5 else 'Clean'
    confidence = avg_prediction * 100 if label == 'Malware' else (1 - avg_prediction) * 100

    return label, confidence

# Function to scan folder and calculate accuracy
def scan_folder(folder_path, expected_label, model):
    total_files = 0
    correct_predictions = 0

    for apk_file in os.listdir(folder_path):
        apk_path = os.path.join(folder_path, apk_file)
        if not os.path.isfile(apk_path):
            continue

        total_files += 1
        label, confidence = predict_apk(apk_path, model)

        if label is not None:
            if (label == 'Malware' and expected_label == 1) or (label == 'Clean' and expected_label == 0):
                correct_predictions += 1

            print(f"APK File: {apk_file}, Prediction: {label}, Confidence: {confidence:.2f}%")

    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0
    return total_files, correct_predictions, accuracy

if __name__ == '__main__':
    model_path = 'malware_classification_model.h5'  # Replace with your model path
    model = load_model(model_path)

    clean_folder = '/home/ubuntu/Downloads/bengin'
    malware_folder = '/home/ubuntu/Downloads/virus'

    # Scan the folders
    clean_total, clean_correct, clean_accuracy = scan_folder(clean_folder, 0, model)
    malware_total, malware_correct, malware_accuracy = scan_folder(malware_folder, 1, model)

    # Calculate overall accuracy
    overall_correct = clean_correct + malware_correct
    overall_total = clean_total + malware_total
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0

    # Print results
    print(f'Clean APKs: {clean_total} scanned, {clean_correct} correct, {clean_accuracy:.2f}% accuracy')
    print(f'Malware APKs: {malware_total} scanned, {malware_correct} correct, {malware_accuracy:.2f}% accuracy')
    print(f'Overall accuracy: {overall_accuracy:.2f}%')
