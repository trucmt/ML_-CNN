import os
import numpy as np
from PIL import Image

def binary_to_image(directory):
    output_dir = '/home/ubuntu/Desktop/anh_clean_ML'
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Step 1: Read binary data
        with open(file_path, 'rb') as binary_file:
            data = binary_file.read()
        
        # Step 2: Convert binary to 8-bit array
        data = bytearray(data)
        np_array = np.array(data, dtype=np.uint8)
        
        # Step 3: Reshape array to form an image
        size = int(len(np_array) ** 0.5)  # Find a suitable size to form a square image
        if size * size != len(np_array):
            # If not a perfect square, add padding or trim excess
            np_array = np_array[:size * size]
        np_array = np_array.reshape((size, size))
        
        # Step 4: Convert to grayscale image and save
        img = Image.fromarray(np_array, 'L')
        try:
            img.save(os.path.join(output_dir, f'image_{filename}.png'))
            print(f"Image saved for {filename}")
        except Exception as e:
            print(f"Failed to save image for {filename}: {e}")

# Usage
if __name__ == "__main__":
    directory = '/home/ubuntu/Downloads/apk_clean/content/drive/MyDrive/Datasets/ApkDatasetSplit/Benign/Benign_2'
    binary_to_image(directory)
