from PIL import Image
import pandas as pd

def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Kiểm tra tính hợp lệ của ảnh
        width, height = img.size
        if width * height > 178956970:  # Giới hạn an toàn mặc định của PIL
            return False
        return True
    except (IOError, SyntaxError, Image.DecompressionBombError) as e:
        return False

# Đọc file CSV
csv_file = 'image_labels.csv'
data_df = pd.read_csv(csv_file)

# Lọc các tệp ảnh không hợp lệ
valid_images = data_df['filepath'].apply(is_valid_image)
data_df = data_df[valid_images]

# Chuyển đổi giá trị cột label thành chuỗi
data_df['label'] = data_df['label'].astype(str)

# Lưu lại DataFrame đã lọc
filtered_csv_file = 'filtered_image_labels.csv'
data_df.to_csv(filtered_csv_file, index=False)
print(f"Filtered CSV file saved to {filtered_csv_file}")
