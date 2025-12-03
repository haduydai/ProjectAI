# BÁO CÁO: PROJECT AI - NHẬN DIỆN CHỮ SỐ VIẾT TAY

## 1. Tổng quan Dự án
Dự án xây dựng hệ thống nhận diện chữ số viết tay (Handwritten Digit Recognition) từ 0 đến 9. Hệ thống bao gồm quy trình huấn luyện mô hình học sâu (Deep Learning) và triển khai ứng dụng web thực tế cho phép người dùng tải ảnh hoặc vẽ trực tiếp để kiểm tra.

* **Mô hình:** LeNet-5 (Kiến trúc CNN cổ điển) vs LSTM (kiến trúc RNN hồi quy)
* **Dữ liệu:** DidaDataset (Ảnh chữ viết tay nền trắng).
* **Công nghệ:** Python, TensorFlow/Keras, OpenCV, Streamlit.

---

## 2. Cơ sở Lý thuyết: Mạng Nơ-ron Tích chập (CNN)

### 2.1. CNN là gì?
CNN (Convolutional Neural Network) là kiến trúc chuyên biệt cho dữ liệu dạng lưới (như hình ảnh). Khác với mạng nơ-ron truyền thống (Dense) thường làm mất thông tin không gian khi duỗi phẳng ảnh, CNN sử dụng các bộ lọc (filters) quét qua ảnh để giữ lại và học các đặc trưng không gian (đường cong, góc cạnh, bố cục).

### 2.2. Các lớp (Layers) thường dùng trong mô hình CNN
Một mô hình CNN điển hình được cấu thành từ việc xếp chồng các loại lớp sau đây:

#### A. Lớp Tích chập (Convolutional Layer - Conv2D)
* **Chức năng:** Đây là "trái tim" của CNN. Nó sử dụng các bộ lọc (Kernels) trượt qua ảnh đầu vào để thực hiện phép tính nhân chập.
* **Tác dụng:** Trích xuất các đặc trưng (Feature Extraction).
    * Ở các lớp đầu: Học các nét đơn giản (đường thẳng, đường chéo, cạnh).
    * Ở các lớp sâu: Học các hình dạng phức tạp hơn (mắt, mũi, hoặc vòng tròn số 0, nét móc số 5).

#### B. Lớp Gộp/Giảm mẫu (Pooling Layer)
* **Chức năng:** Giảm kích thước không gian (chiều rộng x chiều cao) của dữ liệu sau khi đi qua lớp tích chập. Có 2 loại phổ biến: *MaxPooling* (lấy giá trị lớn nhất) và *AveragePooling* (lấy giá trị trung bình).
* **Tác dụng:**
    * Giảm khối lượng tính toán và tham số cho mô hình.
    * Giúp mô hình bất biến với các sai lệch nhỏ (ví dụ: số 5 viết lệch sang trái hay phải một chút thì máy vẫn hiểu là số 5).

#### C. Lớp Hàm kích hoạt (Activation Function)
* **Chức năng:** Đưa tính phi tuyến tính vào mô hình, giúp mạng nơ-ron học được các dữ liệu phức tạp.
* **Các hàm phổ biến:**
    * `ReLU`: Phổ biến nhất hiện nay (nhanh, hiệu quả).
    * `Tanh`: Dùng trong kiến trúc cổ điển (như LeNet-5), đưa giá trị về khoảng [-1, 1].
    * `Softmax`: Thường dùng ở lớp cuối cùng để tính xác suất phân loại.

#### D. Lớp Duỗi phẳng (Flatten Layer)
* **Chức năng:** Là cầu nối giữa phần trích xuất đặc trưng (Conv/Pool) và phần phân loại.
* **Tác dụng:** Chuyển đổi ma trận đặc trưng 3 chiều (Width x Height x Depth) thành một vector 1 chiều dài để đưa vào lớp Dense.

#### E. Lớp Kết nối đầy đủ (Fully Connected / Dense Layer)
* **Chức năng:** Tương tự như mạng nơ-ron truyền thống, nơi mọi nơ-ron lớp này nối với mọi nơ-ron lớp kia.
* **Tác dụng:** Đóng vai trò là bộ phân loại (Classifier). Nó tổng hợp các đặc trưng đã được trích xuất để đưa ra quyết định cuối cùng (ví dụ: Ảnh này 90% là số 8).

---

## 3. Chi tiết Thực hiện: Kiến trúc LeNet-5
Dựa trên lý thuyết trên, dự án này triển khai kiến trúc **LeNet-5** (Yann LeCun, 1998) với cấu hình cụ thể trong `train.py` như sau: 
| Thứ tự | Loại Layer | Cấu hình chi tiết | Vai trò trong dự án |
| :--- | :--- | :--- | :--- |
| **1** | **Input** | 32x32x1 | Ảnh xám đầu vào. |
| **2** | **Conv2D** | 6 filters, 5x5, Tanh | Trích xuất nét cơ bản. |
| **3** | **AveragePooling** | 2x2, stride=2 | Giảm kích thước ảnh còn 14x14. |
| **4** | **Conv2D** | 16 filters, 5x5, Tanh | Trích xuất hình dạng phức tạp. |
| **5** | **AveragePooling** | 2x2, stride=2 | Giảm kích thước ảnh còn 5x5. |
| **6** | **Flatten** | - | Duỗi ma trận 5x5x16 thành vector. |
| **7** | **Dense** | 120 units, Tanh | Lớp ẩn xử lý thông tin. |
| **8** | **Dense** | 84 units, Tanh | Lớp ẩn xử lý thông tin. |
| **9** | **Output (Dense)** | 10 units, Softmax | Trả về xác suất cho 10 số (0-9). |

---

## 4. Quy trình Xây dựng Đồ án
Quy trình thực hiện được chia làm 5 giai đoạn chính:

### Bước 1: Thu thập & Môi trường
* Dữ liệu: **DidaDataset** (250.000 ảnh).
* Công cụ: Thiết lập môi trường ảo (`venv`), cài đặt thư viện `TensorFlow`, `OpenCV`.

### Bước 2: Tiền xử lý dữ liệu (Preprocessing)
Xử lý logic trong `data_loader.py` và `train.py` để đồng bộ dữ liệu thực tế với tư duy của máy:
1.  **Grayscale:** Chuyển ảnh màu sang ảnh xám.
2.  **Resize:** Đưa về kích thước 32x32.
3.  **Invert Colors (Quan trọng):** Đảo ngược màu ảnh (Nền trắng chữ đen -> Nền đen chữ trắng) để khớp với chuẩn MNIST.
4.  **Normalize:** Chia giá trị pixel cho 255.0 để đưa về [0, 1].

### Bước 3: Huấn luyện (Training)
* **Optimizer:** Adam.
* **Loss Function:** Sparse Categorical Crossentropy.
* **Epochs:** 10 vòng lặp.
* **Batch Size:** 128 (Tối ưu tốc độ).

### Bước 4: Đánh giá & Lưu model
* Kiểm tra độ chính xác trên tập Test (20% dữ liệu).
* Lưu trọng số model vào file `.h5`.

### Bước 5: Ứng dụng Web (Streamlit)
* Xây dựng giao diện với 2 tính năng: Upload ảnh và Vẽ trực tiếp (Canvas).
* Hiển thị biểu đồ độ tin cậy của dự đoán AI.

---

## 5. Hướng dẫn Cài đặt & Sử dụng

### Yêu cầu hệ thống
* Python 3.7+
* Thư viện: TensorFlow, OpenCV, Streamlit, NumPy.

### Các bước chạy dự án
1.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    pip install streamlit-drawable-canvas
    ```

2.  **Huấn luyện mô hình:**
    ```bash
    python train.py
    ```
    *(Sẽ tạo ra file `models/digit_model.h5`)*

3.  **Chạy ứng dụng Web:**
    ```bash
    streamlit run app.py
    ```

---

## 6. Tổng hợp câu lệnh (Cheat Sheet)

* **Tạo môi trường ảo:** `python -m venv venv`
* **Kích hoạt (Win):** `.\venv\Scripts\activate`
* **Cài thư viện:** `pip install -r requirements.txt`
* **Train:** `python train.py`
* **Chạy Web:** `streamlit run app.py`
* **Git Push:**
    ```bash
    git add .
    git commit -m "Update project"
    git push origin main
    ```

---
*---------------------------------end--------------------------------------*
* tham khảo:
* https://gemini.google.com/
* https://colah.github.io/
* https://www.kaggle.com/code/blurredmachine/lenet-architecture-a-complete-guide
* https://www.youtube.com/watch?v=S3d_Hp3UiJI&t=352s
* https://www.youtube.com/shorts/N6NBT-n9mmo
* https://www.youtube.com/watch?v=WMC7_kvsrZg
* https://www.youtube.com/watch?v=sWPNm_GhhCA&t=241s
* https://www.youtube.com/watch?v=pj9-rr1wDhM
