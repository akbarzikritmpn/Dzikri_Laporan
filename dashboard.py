import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Muhammad Akbar Dzikri_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Zikri_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# CSS
# ==========================
custom_style = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #d6edc7, #95bfa1);
    color: #2d4739;
    font-family: 'Arial', sans-serif;
}

/* JUDUL */
.title-box {
    background: linear-gradient(145deg, #7ba883, #547a64);
    padding: 15px;
    border-radius: 15px;
    border: 2px solid #c9e7c0;
    text-align: center;
    color: #d6edc7;
    font-size: 32px;
    font-weight: bold;
    box-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    margin-bottom: 30px;
}

/* LAYOUT DUA KOLOM */
.main-container {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: flex-start;
    gap: 40px;
}

/* KIRI */
.left-box {
    width: 35%;
    background: linear-gradient(145deg, #7ba883, #547a64);
    padding: 25px;
    border-radius: 15px;
    border: 2px solid #c9e7c0;
    color: #d6edc7;
    box-shadow: 3px 3px 6px rgba(0,0,0,0.25);
}

/* Upload box */
div[data-testid="stFileUploader"] {
    background: #7ba883;
    border: 2px dashed #c9e7c0;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    color: #f0f8ec !important;
}

div[data-testid="stFileUploader"] section {
    color: #f0f8ec !important;
}

/* KANAN */
.right-box {
    width: 60%;
    background: linear-gradient(145deg, #7ba883, #547a64);
    border: 2px solid #c9e7c0;
    border-radius: 15px;
    text-align: center;
    padding: 20px;
    color: #d6edc7;
    box-shadow: 3px 3px 6px rgba(0,0,0,0.25);
}

/* Label */
.box-label {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 15px;
    text-align: center;
    color: #d6edc7;
}

/* Gambar */
.image-preview {
    border-radius: 10px;
    border: 2px solid #c9e7c0;
    margin-top: 10px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
}

/* Kotak hasil prediksi */
.result-box {
    background: #6f9b7c;
    border: 2px solid #c9e7c0;
    border-radius: 10px;
    margin-top: 15px;
    padding: 10px;
    color: #eaf4e2;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
    font-weight: bold;
    text-align: center;
}
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# ==========================
# JUDUL
# ==========================
st.markdown('<div class="title-box">🧠 DETEKSI & KLASIFIKASI GAMBAR</div>', unsafe_allow_html=True)

# ==========================
# KONTEN DUA KOLOM
# ==========================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ==========================
# KOLOM KIRI
# ==========================
st.markdown('<div class="left-box">', unsafe_allow_html=True)
st.markdown('<div class="box-label">PENGATURAN INPUT</div>', unsafe_allow_html=True)

# Pilih mode di dalam kotak
menu = st.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"], label_visibility="visible")

# Upload gambar di bawahnya
uploaded_file = st.file_uploader("Seret atau pilih gambar di sini 👇", type=["jpg", "jpeg", "png"])

st.markdown('</div>', unsafe_allow_html=True)  # tutup left-box

# ==========================
# KOLOM KANAN (HASIL)
# ==========================
st.markdown('<div class="right-box">', unsafe_allow_html=True)
st.markdown('<div class="box-label">HASIL DETEKSI / KLASIFIKASI</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True, output_format="auto")

    if menu == "Deteksi Objek (YOLO)":
        img_array = np.array(img)
        results = yolo_model(img_array)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True, output_format="auto")

        st.markdown(f'<div class="result-box">✅ Deteksi objek berhasil dilakukan.</div>', unsafe_allow_html=True)

    elif menu == "Klasifikasi Gambar":
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        accuracy = float(np.max(prediction)) * 100

        st.markdown(
            f'<div class="result-box">📊 <b>Hasil Prediksi:</b> {class_index}<br>🎯 <b>Akurasi:</b> {accuracy:.2f}%</div>',
            unsafe_allow_html=True
        )
else:
    st.info("Silakan unggah gambar terlebih dahulu di sebelah kiri.")

st.markdown('</div>', unsafe_allow_html=True)  # tutup right box
st.markdown('</div>', unsafe_allow_html=True)  # tutup main container
