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

/* LAYOUT */
.main-container {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: flex-start;
    gap: 30px;
}

/* KIRI */
.left-box {
    width: 30%;
    background: linear-gradient(145deg, #7ba883, #547a64);
    padding: 25px;
    border-radius: 15px;
    border: 2px solid #c9e7c0;
    color: #d6edc7;
    box-shadow: 3px 3px 6px rgba(0,0,0,0.25);
}

/* Upload box di dalam kotak hijau */
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
    width: 65%;
    background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
    border: 2px solid #c9e7c0;
    border-radius: 15px;
    text-align: center;
    padding: 20px;
    box-shadow: 3px 3px 6px rgba(0,0,0,0.25);
}

/* Label */
.box-label {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
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
# KONTEN
# ==========================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# KOTAK KIRI
st.markdown('<div class="left-box">', unsafe_allow_html=True)
st.markdown('<div class="box-label">UPLOAD GAMBAR</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Seret atau pilih gambar di sini 👇", type=["jpg", "jpeg", "png"])
menu = st.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"], label_visibility="visible")

st.markdown('</div>', unsafe_allow_html=True)  # tutup left-box

# KOTAK KANAN
st.markdown('<div class="right-box">', unsafe_allow_html=True)

if uploaded_file is not None:
    st.markdown('<div class="box-label">GAMBAR</div>', unsafe_allow_html=True)
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        img_array = np.array(img)
        results = yolo_model(img_array)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.success(f"Hasil Prediksi: {class_index}")
        st.info(f"Probabilitas: {np.max(prediction):.4f}")
else:
    st.info("Silakan unggah gambar terlebih dahulu di sebelah kiri.")

st.markdown('</div>', unsafe_allow_html=True)  # tutup right box
st.markdown('</div>', unsafe_allow_html=True)  # tutup main container
