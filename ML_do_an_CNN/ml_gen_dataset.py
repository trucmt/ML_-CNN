import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

# Đọc file CSV đã lọc
csv_file = 'filtered_image_labels.csv'
data_df = pd.read_csv(csv_file)

# Chuyển đổi giá trị cột label thành chuỗi
data_df['label'] = data_df['label'].astype(str)

# Cài đặt các tham số
img_size = 64
batch_size = 32

# Trình tạo dữ liệu
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Trình tạo dữ liệu huấn luyện
train_generator = datagen.flow_from_dataframe(
    dataframe=data_df,
    x_col='filepath',
    y_col='label',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

# Trình tạo dữ liệu kiểm tra
validation_generator = datagen.flow_from_dataframe(
    dataframe=data_df,
    x_col='filepath',
    y_col='label',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Tải mô hình VGG16 và thêm các lớp đầu ra
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Đóng băng các lớp của VGG16
for layer in base_model.layers:
    layer.trainable = False

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Đánh giá mô hình
loss, accuracy = model.evaluate(validation_generator)
print(f"Độ chính xác trên tập kiểm tra: {accuracy}")

# Lưu mô hình
model.save('malware_classification_model.h5')
