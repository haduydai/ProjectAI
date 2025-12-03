import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Cấu hình trang Web
st.set_page_config(
    page_title="Dự án Nhận diện số viết tay",
    page_icon="🤖",
    layout="centered"
)

# 2. Hàm load model (Dùng Cache để không phải load lại mỗi lần f5)
@st.cache_resource
def load_my_model():
    model_path = 'models/digit_model.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        return None

# 3. Giao diện chính
st.title("🤖 Demo Nhận Diện Số Viết Tay")
st.write("Mô hình: **LeNet-5** | Dữ liệu: **DIDADATASET**")
st.write("---")

# Load model
model = load_my_model()

if model is None:
    st.error("❌ Không tìm thấy file 'models/digit_model.h5'. Hãy chạy file train.py trước!")
else:
    # 4. Khu vực upload ảnh
    uploaded_file = st.file_uploader("📤 Tải ảnh chứa số (0-9) lên đây:", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Chia cột để hiển thị ảnh gốc và ảnh sau xử lý
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("📸 **Ảnh gốc:**")
            # Mở ảnh bằng thư viện PIL
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, use_container_width=True)

        # --- XỬ LÝ ẢNH (Preprocessing) ---
        # Bước A: Chuyển sang ảnh xám (Grayscale)
        # Convert PIL -> Numpy array
        img_array = np.array(image_pil.convert('L'))

        # Bước B: Resize về 32x32 (Đúng chuẩn LeNet-5)
        # Dùng OpenCV để resize chất lượng tốt hơn
        img_resized = cv2.resize(img_array, (32, 32))

        # Bước C: Đảo màu (Quan trọng!)
        # AI học trên nền đen chữ trắng. Nếu ảnh tải lên là nền trắng chữ đen (giấy viết), ta phải đảo ngược.
        # Logic: Nếu độ sáng trung bình > 127 (tức là ảnh sáng/nền trắng) -> Đảo.
        if np.mean(img_resized) > 127:
            img_resized = 255 - img_resized

        # Bước D: Chuẩn hóa pixel về [0, 1] và Reshape
        img_input = img_resized / 255.0
        img_input = img_input.reshape(1, 32, 32, 1)

        with col2:
            st.write("🧠 **AI nhìn thấy:**")
            st.image(img_resized, caption="32x32 px (Đã đảo màu)", width=150)

        # 5. Nút Dự đoán
        if st.button("🔍 DỰ ĐOÁN NGAY", type="primary"):
            with st.spinner('AI đang suy nghĩ...'):
                # Model dự đoán
                prediction = model.predict(img_input)
                
                # Lấy kết quả cao nhất
                ket_qua = np.argmax(prediction)
                do_chinh_xac = np.max(prediction) * 100
                
            # Hiển thị kết quả
            st.success(f"Kết quả: **SỐ {ket_qua}**")
            st.info(f"Độ tự tin: **{do_chinh_xac:.2f}%**")
            
            # Vẽ biểu đồ xác suất
            st.write("Biểu đồ xác suất:")
            st.bar_chart(prediction[0])

# Footer
st.markdown("---")
st.caption("Developed by Ha Duy Dai")