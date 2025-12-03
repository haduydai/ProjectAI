import os
import tensorflow as tf
from tensorflow.keras import layers, models
from data_loader import DidaLoader 

# --- CẤU HÌNH TỐI ƯU ---
# Trỏ vào thư mục chứa dataset
DATASET_PATH = os.path.join('dataset', '250000', '250000_Final')

# 1. Giới hạn số lượng ảnh (hoặc None nếu máy mạnh)
MAX_IMAGES = 50000  

# 2. Tăng Batch Size (Mặc định 32 -> Tăng lên 128 hoặc 256)
# Số càng to train càng nhanh, nhưng tốn VRAM. Nếu lỗi OOM thì giảm xuống 64.
BATCH_SIZE = 128 

# 3. Số vòng lặp (Epochs)
EPOCHS = 10

MODEL_PATH = 'models/digit_model.h5'

# --- KIỂM TRA & CẤU HÌNH GPU ---
print("\n" + "="*40)
print("🔍 ĐANG KIỂM TRA PHẦN CỨNG...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Đã phát hiện {len(gpus)} GPU: {gpus}")
    try:
        # Cấu hình để GPU không bị chiếm dụng 100% bộ nhớ ngay lập tức
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🚀 Đã kích hoạt chế độ tối ưu bộ nhớ GPU!")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ Không tìm thấy GPU. Code sẽ chạy bằng CPU (chậm hơn).")
print("="*40 + "\n")


# --- XÂY DỰNG MÔ HÌNH ---
def build_lenet5():
    print("🛠️ Đang xây dựng mô hình LeNet-5...")
    model = models.Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(6, (5, 5), activation='tanh'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(16, (5, 5), activation='tanh'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# --- MAIN ---
if __name__ == "__main__":
    # 1. Load dữ liệu
    loader = DidaLoader(data_path=DATASET_PATH, max_images=MAX_IMAGES)
    (x_train, y_train), (x_test, y_test) = loader.load()

    # 2. Khởi tạo mô hình
    model = build_lenet5()
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # 3. Huấn luyện (Đã thêm batch_size)
    print(f"🚀 Bắt đầu huấn luyện với Batch Size = {BATCH_SIZE}...")
    
    model.fit(
        x_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,  # <--- Tăng tốc độ trainS
        validation_data=(x_test, y_test)
    )

    # 4. Lưu kết quả
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save(MODEL_PATH)
    print(f"\n🎉 HOÀN THÀNH! Mô hình đã lưu tại: {MODEL_PATH}")
