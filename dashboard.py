import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ====== Konfigurasi halaman agar lebar penuh ======
st.set_page_config(layout="wide")

# ====== Load Model ======
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Muhammad Akbar Dzikri_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Zikri_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ====== CSS Styling ======
st.markdown("""
<style>
/* ====== Full width container ====== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #d6edc7, #95bfa1);
    color: #2d4739;
    font-family: 'Arial', sans-serif;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

/* ====== Kotak judul utama ====== */
.main-title {
    background: linear-gradient(145deg, #6b9474, #547a64);
    border: 3px solid #c9e7c0;
    border-radius: 20px;
    color: #eaf4e2;
    text-align: center;
    padding: 20px;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 25px;
    box-shadow: 4px 4px 8px rgba(0,0,0,0.25);
    width: 100%;
}

/* ====== Kotak hijau untuk bagian 'Pilih Mode' & 'Upload' ====== */
.section-box {
    background: linear-gradient(145deg, #7ba883, #547a64);
    border-radius: 20px;
    border: 2px solid #c9e7c0;
    padding: 25px;
    color: #d6edc7;
    box-shadow: 4px 4px 8px rgba(0,0,0,0.25);
}

/* ====== Judul dalam kotak ====== */
.section-title {
    font-size: 22px;
    font-weight: bold;
    background-color: #6b9474;
    padding: 8px 15px;
    border-radius: 12px;
    color: #eaf4e2;
    margin-bottom: 15px;
    text-align: center;
    border: 2px solid #c9e7c0;
}

/* ====== Upload box ====== */
div[data-testid="stFileUploader"] {
    background: #7ba883;
    border: 2px dashed #c9e7c0;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    color: #f0f8ec !important;
}

/* ====== Hasil deteksi / klasifikasi ====== */
.detect-result {
    background: #6f9b7c;
    border: 2px solid #c9e7c0;
    border-radius: 10px;
    margin-top: 15px;
    padding: 10px;
    color: #eaf4e2;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ====== Judul utama ======
st.markdown('<div class="main-title">🧠 Deteksi dan Klasifikasi Gambar</div>', unsafe_allow_html=True)

# ====== Upload gambar di halaman utama ======
uploaded_file = st.file_uploader("Seret atau pilih gambar (unggah sekali saja) 👇", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

    # ====== Sidebar navigation ======
    page = st.sidebar.radio("⚙️ Pilih Mode Analisis", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

    # Konversi ke array numpy untuk deteksi / klasifikasi
    img_array = np.array(img)

    # ====== Halaman Deteksi Objek ======
    if page == "Deteksi Objek (YOLO)":
        st.markdown('<div class="section-title">📦 Hasil Deteksi Objek</div>', unsafe_allow_html=True)
        results = yolo_model(img_array)

        # Copy gambar untuk digambar kotak deteksi
        img_with_boxes = img_array.copy()

        # Loop tiap bounding box hasil deteksi
        for box in results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = int(box.cls[0])
            # Gambar kotak hijau
            cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # Tulis label & confidence
            cv2.putText(img_with_boxes, f"{label} {confidence:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        st.image(img_with_boxes, caption="📦 Gambar dengan Bounding Box Deteksi", use_container_width=True)
        st.markdown('<div class="detect-result">✅ Deteksi objek berhasil dilakukan.</div>', unsafe_allow_html=True)

    # ====== Halaman Klasifikasi Gambar ======
    elif page == "Klasifikasi Gambar":
        st.markdown('<div class="section-title">📊 Hasil Klasifikasi Gambar</div>', unsafe_allow_html=True)

        img_resized = img.resize((224, 224))
        img_array_cls = image.img_to_array(img_resized)
        img_array_cls = np.expand_dims(img_array_cls, axis=0)
        img_array_cls = img_array_cls / 255.0

        prediction = classifier.predict(img_array_cls)
        class_index = np.argmax(prediction)
        accuracy = float(np.max(prediction)) * 100

        st.markdown(
            f'<div class="detect-result">📊 <b>Hasil Prediksi:</b> {class_index}<br>🎯 <b>Akurasi:</b> {accuracy:.2f}%</div>',
            unsafe_allow_html=True
        )
else:
    st.info("Silakan unggah gambar terlebih dahulu.")
