# Import các thư viện
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
# Import các thư viện
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
# Thiết lập tham số
IMG_SIZE = (380, 380)
BATCH_SIZE = 16
NUM_CLASSES = 20
EPOCHS = 40
DATASET_PATH = '/content/drive/MyDrive/Medicine Chinese/NB-TCM-CHM/Dataset 2'  # Thay đường dẫn đến thư mục chứa dataset

# Hàm tiền xử lý ảnh
def preprocess_image(image_path, target_size=IMG_SIZE):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    if image_array.shape[-1] == 4:  # Kiểm tra nếu ảnh có 4 kênh (RGBA) thì chuyển sang RGB
        image_array = image_array[..., :3]
    image_array = image_array.astype("float32") / 255.0
    return image_array
# Load dữ liệu từ thư mục
def load_data(data_dir):
    inputs = []
    targets = []
    class_names = os.listdir(data_dir)
    class_names.sort()  # Sắp xếp tên lớp để có thứ tự cố định
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):  # Kiểm tra nếu là thư mục
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                try:
                    image_array = preprocess_image(image_path)
                    inputs.append(image_array)
                    targets.append(class_index)
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
    return np.array(inputs), np.array(targets), class_names

# Load dữ liệu
inputs, targets, class_names = load_data(DATASET_PATH)
print(f"Loaded {len(inputs)} images with {len(class_names)} classes.")
# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Chuyển đổi nhãn thành one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_one_hot = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
# Xây dựng mô hình sử dụng EfficientNetB4
def build_model(input_shape=(380, 380, 3), num_classes=20):
    # Tải mô hình EfficientNetB4, loại bỏ lớp phân loại phía trên
    base_model = EfficientNetB4(weights="imagenet", include_top=False, input_shape=input_shape)

    # Thêm các lớp tùy chỉnh vào trên mô hình gốc
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Lớp Pooling để giảm chiều dữ liệu
    x = Dropout(0.5)(x)  # Lớp Dropout để ngăn chặn overfitting
    x = Dense(1024, activation='relu')(x)  # Lớp kết nối đầy đủ với hàm kích hoạt ReLU
    x = Dropout(0.5)(x)  # Thêm một lớp Dropout khác

    # Lớp đầu ra cho phân loại
    outputs = Dense(num_classes, activation='softmax')(x)

    # Tạo mô hình đầy đủ
    model = Model(inputs=base_model.input, outputs=outputs)

    # Biên dịch mô hình với optimizer và hàm mất mát phù hợp
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',  # Đổi thành 'categorical_crossentropy'
                  metrics=['accuracy'])

    return model

# Xây dựng mô hình
model = build_model()
model.summary()
# Tạo bộ dữ liệu ImageDataGenerator cho tập huấn luyện và tập validation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Tạo generator cho tập huấn luyện
train_generator = train_datagen.flow(
    X_train, y_train_one_hot,
    batch_size=BATCH_SIZE,
    shuffle=True  # Shuffle dữ liệu huấn luyện
)

# Tạo generator cho tập validation
validation_generator = val_datagen.flow(
    X_val, y_val_one_hot,
    batch_size=BATCH_SIZE,
    shuffle=False  # Không cần shuffle tập validation
)

print(f"Train generator batches: {len(train_generator)}")
print(f"Validation generator batches: {len(validation_generator)}")

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    verbose=1
)
# Đánh giá mô hình
val_loss, val_accuracy = model.evaluate(X_val, y_val_one_hot, verbose=1)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Dự đoán và báo cáo
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val_one_hot, axis=1)

print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

# Ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()