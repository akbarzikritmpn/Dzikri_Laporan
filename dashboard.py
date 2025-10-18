import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ====== Load Model ======
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Muhammad Akbar Dzikri_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Zikri_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ====== CSS ======
st.markdown("""
<style>
/* ==== Fullscreen Layout ==== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #d6edc7, #95bfa1);
    color: #2d4739;
    font-family: 'Arial', sans-serif;
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stHeader"] {
    display: none;
}
[data-testid="stToolbar"] {
    display: none;
}
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* ====== Halaman Awal CSS ====== */
.welcome-box {
    background: linear-gradient(145deg, #57876a, #9dbcae);
    border-radius: 12px;
    padding: 20px 30px;
    margin-bottom: 30px;
    text-align: center;
    font-weight: bold;
    font-size: 22px;
    border: 1.5px solid #c9e7c0;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
    width: 100%;
    max-width: 100vw;
    box-sizing: border-box;
}

.main-box {
    background: linear-gradient(145deg, #8daaa5, #618472);
    border-radius: 15px;
    padding: 70px 40px;
    margin-bottom: 40px;
    font-weight: bold;
    font-size: 34px;
    text-align: center;
    border: 2px solid #c9e7c0;
    box-shadow: 4px 4px 12px rgba(0,0,0,0.3);
    line-height: 1.4;
    width: 100%;
    max-width: 100vw;
    box-sizing: border-box;
}

/* Tombol halaman awal */
.stButton > button {
    width: 180px;
    height: 45px;
    font-weight: bold;
    font-size: 14px;
    color: #fff;
    background-color: #547a64;
    border-radius: 12px;
    border: 1.5px solid #c9e7c0;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #7ba883;
}

/* ====== Kotak Pilih Mode & Upload ====== */
.section-box {
    background: linear-gradient(145deg, #7ba883, #547a64);
    border-radius: 20px;
    border: 2px solid #c9e7c0;
    padding: 25px;
    color: #d6edc7;
    box-shadow: 4px 4px 8px rgba(0,0,0,0.25);
    width: 100%;
}

/* ====== Judul Kecil dalam Kotak ====== */
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

/* ====== Upload Box ====== */
div[data-testid="stFileUploader"] {
    background: #7ba883;
    border: 2px dashed #c9e7c0;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    color: #f0f8ec !important;
}

/* ====== Hasil Deteksi/Klasifikasi ====== */
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

# ====== Halaman Awal ======
def halaman_awal():
    st.markdown('<div class="welcome-box">SELAMAT DATANG DI DASHBOARD MUHAMMAD AKBAR DZIKRI</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="main-box">
    KLASIFIKASI GAMBAR <br> & <br> OBJEK DETECTION
    </div>
    """, unsafe_allow_html=True)

    if st.button("HALAMAN BERIKUTNYA"):
        st.session_state['page'] = 'main'

# ====== Halaman Utama (Kode yang kamu punya) ======
def halaman_utama():
    st.markdown('<div class="main-title">🧠 Deteksi dan Klasifikasi Gambar</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-title">⚙️ Pilih Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">📤 Upload & Hasil Deteksi / Klasifikasi</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Seret atau pilih gambar di sini 👇", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

            if mode == "Deteksi Objek (YOLO)":
                img_array = np.array(img)
                results = yolo_model(img_array)
                img_with_boxes = img_array.copy()
                class_names = yolo_model.names

                for box in results[0].boxes:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    label_index = int(box.cls[0])
                    label_name = class_names[label_index]

                    cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"{label_name} {confidence:.2f}",
                                (xmin, max(ymin - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)

                st.image(img_with_boxes, caption="📦 Hasil Deteksi dengan Bounding Box", use_container_width=True)
                st.markdown('<div class="detect-result">✅ Deteksi objek berhasil dilakukan.</div>', unsafe_allow_html=True)

            elif mode == "Klasifikasi Gambar":
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                accuracy = float(np.max(prediction)) * 100

                class_labels = ["Kelas 1", "Kelas 2", "Kelas 3", "Kelas 4", "Kelas 5"]
                class_name = class_labels[class_index] if class_index < len(class_labels) else str(class_index)

                st.markdown(
                    f'<div class="detect-result">📊 <b>Hasil Prediksi:</b> {class_name}<br>🎯 <b>Akurasi:</b> {accuracy:.2f}%</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Silakan unggah gambar terlebih dahulu di atas.")
        st.markdown('</div>', unsafe_allow_html=True)

# ====== Main Program ======
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

if st.session_state['page'] == 'home':
    halaman_awal()
elif st.session_state['page'] == 'main':
    halaman_utama()
