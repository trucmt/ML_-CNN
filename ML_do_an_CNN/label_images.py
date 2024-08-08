import os
import pandas as pd
import numpy as np

# Đặt đường dẫn tới thư mục chứa ảnh
clean_image_dir = '/home/ubuntu/Desktop/anh_clean_ML'
malware_image_dir = '/home/ubuntu/Desktop/anh_dataset_Ml'
output_csv = 'image_labels.csv'

# Hàm tạo DataFrame từ các tệp ảnh và nhãn
def create_label_dataframe(clean_dir, malware_dir):
    data = []
    # Duyệt qua thư mục ảnh sạch và thêm vào danh sách với nhãn 0
    for filename in os.listdir(clean_dir):
        if filename.endswith('.png'):  # Kiểm tra xem tệp có phải là ảnh PNG không
            file_path = os.path.join(clean_dir, filename)
            data.append([file_path, 0])
    
    # Duyệt qua thư mục ảnh malware và thêm vào danh sách với nhãn 1
    for filename in os.listdir(malware_dir):
        if filename.endswith('.png'):  # Kiểm tra xem tệp có phải là ảnh PNG không
            file_path = os.path.join(malware_dir, filename)
            data.append([file_path, 1])
    
    # Tạo DataFrame từ danh sách
    df = pd.DataFrame(data, columns=['filepath', 'label'])
    return df

# Tạo DataFrame và lưu vào file CSV
df = create_label_dataframe(clean_image_dir, malware_image_dir)

# Trộn ngẫu nhiên các hàng trong DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Lưu DataFrame đã trộn vào file CSV
df.to_csv(output_csv, index=False)
print(f"CSV file saved to {output_csv}")
