import cv2
import os
import numpy as np
from sklearn.utils import shuffle

class DidaLoader:
    def __init__(self, data_path='dataset', img_size=32, max_images=1000):
        """
        Khởi tạo bộ load dữ liệu.
        :param data_path: Đường dẫn thư mục chứa dữ liệu (dataset/0, dataset/1...)
        :param img_size: Kích thước ảnh chuẩn hóa (mặc định 32x32)
        :param max_images: Số lượng ảnh tối đa load mỗi folder (để test nhanh)
        """
        self.data_path = data_path
        self.img_size = img_size
        self.max_images = max_images

    def preprocess_image(self, img_path):
        """Đọc và xử lý một ảnh: Đọc -> Xám -> Resize -> Đảo màu"""
        try:
            # Đọc ảnh
            img = cv2.imread(img_path)
            if img is None: return None

            # 1. Chuyển sang ảnh xám
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 2. Resize
            img = cv2.resize(img, (self.img_size, self.img_size))

            # 3. Đảo màu (Nếu nền trắng chữ đen -> Đổi thành nền đen chữ trắng)
            # DIDADATASET là chữ viết tay trên giấy trắng, nên cần đảo ngược để giống MNIST
            if np.mean(img) > 127:
                img = 255 - img
            
            return img
        except Exception as e:
            return None

    def load(self):
        """Hàm chính để load toàn bộ dữ liệu"""
        print(f"🔄 Đang khởi tạo DidaLoader từ: {os.path.abspath(self.data_path)}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ LỖI: Không tìm thấy thư mục '{self.data_path}'")

        images = []
        labels = []

        # Duyệt qua các folder 0-9
        for i in range(10):
            folder_path = os.path.join(self.data_path, str(i))
            if not os.path.exists(folder_path):
                continue

            print(f"   - Đang xử lý số {i}...", end=" ")
            count = 0
            
            for filename in os.listdir(folder_path):
                if self.max_images and count >= self.max_images:
                    break

                img_path = os.path.join(folder_path, filename)
                img = self.preprocess_image(img_path)

                if img is not None:
                    images.append(img)
                    labels.append(i)
                    count += 1
            
            print(f"-> Xong {count} ảnh.")

        # Kiểm tra dữ liệu
        if len(images) == 0:
            raise ValueError("❌ Không load được ảnh nào!")

        # Chuyển sang Numpy array
        X = np.array(images)
        y = np.array(labels)

        # Xáo trộn dữ liệu
        X, y = shuffle(X, y, random_state=42)

        # Chuẩn hóa về [0, 1] và Reshape (N, 32, 32, 1)
        X = X / 255.0
        X = X.reshape(-1, self.img_size, self.img_size, 1)

        # Chia Train/Test (80% - 20%)
        split = int(len(X) * 0.8)
        x_train, x_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"✅ Đã load xong: {len(x_train)} Train | {len(x_test)} Test")
        return (x_train, y_train), (x_test, y_test)