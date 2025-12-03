import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas # Thư viện mới

# 1. Cấu hình trang Web
st.set_page_config(
    page_title="Dự án Nhận diện số viết tay",
    page_icon="🤖",
    layout="centered"
)

# 2. Hàm load model
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
st.write("Mô hình: **LeNet-5** | Dữ liệu: **DIDADATASET**"),

# Load model
model = load_my_model()

if model is None:
    st.error("❌ Không tìm thấy file 'models/digit_model.h5'. Hãy chạy file train.py trước!")
    st.stop() # Dừng lại nếu không có model

# --- TẠO 2 TAB CHỨC NĂNG ---
tab1, tab2 = st.tabs(["📤 Tải ảnh lên", "✍️ Vẽ trực tiếp"])

# ================= TAB 1: UPLOAD ẢNH (Code cũ) =================
with tab1:
    uploaded_file = st.file_uploader("Tải ảnh chứa số (0-9):", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2),
        
        with col1:
            st.write("📸 **Ảnh gốc:**"),
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, use_container_width=True)

        # XỬ LÝ ẢNH
        img_array = np.array(image_pil.convert('L')),
        img_resized = cv2.resize(img_array, (32, 32))

        # Đảo màu nếu ảnh là nền trắng chữ đen
        if np.mean(img_resized) > 127:
            img_resized = 255 - img_resized

        img_input = img_resized / 255.0,
        img_input = img_input.reshape(1, 32, 32, 1)

        with col2:
            st.write("🧠 **AI nhìn thấy:**")
            st.image(img_resized, caption="32x32 px", width=150)

        # NÚT DỰ ĐOÁN
        if st.button("🔍 DỰ ĐOÁN (Upload)", type="primary"):
            prediction = model.predict(img_input)
            ket_qua = np.argmax(prediction)
            do_chinh_xac = np.max(prediction) * 100
            
            st.success(f"Kết quả: **SỐ {ket_qua}**")
            st.info(f"Độ tự tin: **{do_chinh_xac:.2f}%**")
            st.bar_chart(prediction[0])


# ================= TAB 2: VẼ SỐ (Tính năng mới) =================
with tab2:
    st.write("Vẽ số vào khung bên dưới:")
    
    # Tạo Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=15,      # Nét vẽ to một chút để khi resize không bị mất
        stroke_color="#FFFFFF", # Bút màu TRẮNG
        background_color="#000000", # Nền ĐEN (AI thích nền đen chữ trắng)
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("🔍 DỰ ĐOÁN (Hình vẽ)", type="primary"):
        if canvas_result.image_data is not None:
            # Lấy dữ liệu ảnh từ Canvas
            img_data = canvas_result.image_data.astype('uint8')
            
            # Canvas trả về RGBA -> Chuyển sang Grayscale
            img_gray = cv2.cvtColor(img_data, cv2.COLOR_RGBA2GRAY)
            
            # Resize về 32x32
            img_resized = cv2.resize(img_gray, (32, 32))
            
            # Lưu ý: Vì ta vẽ bút trắng nền đen nên KHÔNG CẦN ĐẢO MÀU nữa
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("Kích thước thực:")
                st.image(img_resized, caption="32x32 Input", width=100)
            
            # Chuẩn hóa
            img_input = img_resized / 255.0
            img_input = img_input.reshape(1, 32, 32, 1)
            
            # Dự đoán
            prediction = model.predict(img_input)
            ket_qua = np.argmax(prediction)
            do_chinh_xac = np.max(prediction) * 100
            
            with col_b:
                st.success(f"Kết quả: **SỐ {ket_qua}**")
                st.write(f"Độ chính xác: {do_chinh_xac:.1f}%")
            
            st.bar_chart(prediction[0])

# Footer
st.markdown("---"),
