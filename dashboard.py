import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

st.set_page_config(layout="wide")

@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Muhammad Akbar Dzikri_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Zikri_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

st.markdown("""
<style>
/* ... CSS styling sama seperti sebelumnya ... */
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 Deteksi dan Klasifikasi Gambar</div>', unsafe_allow_html=True)

# Navigation menu di sidebar
page = st.sidebar.radio("Pilih Halaman", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

if page == "Deteksi Objek (YOLO)":
    st.markdown('<div class="section-title">📤 Upload & Hasil Deteksi</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Seret atau pilih gambar untuk deteksi 👇", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        img_array = np.array(img)
        results = yolo_model(img_array)

        img_with_boxes = img_array.copy()
        for box in results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = int(box.cls[0])
            cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"{label} {confidence:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        st.image(img_with_boxes, caption="📦 Hasil Deteksi", use_container_width=True)
        st.markdown('<div class="detect-result">✅ Deteksi objek berhasil dilakukan.</div>', unsafe_allow_html=True)
    else:
        st.info("Silakan unggah gambar terlebih dahulu.")

elif page == "Klasifikasi Gambar":
    st.markdown('<div class="section-title">📤 Upload & Hasil Klasifikasi</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Seret atau pilih gambar untuk klasifikasi 👇", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        accuracy = float(np.max(prediction)) * 100

        st.markdown(
            f'<div class="detect-result">📊 <b>Hasil Prediksi:</b> {class_index}<br>🎯 <b>Akurasi:</b> {accuracy:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Silakan unggah gambar terlebih dahulu.")
