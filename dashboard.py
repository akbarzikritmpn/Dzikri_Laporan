import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2  # ✅ Tambahan

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
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #d6edc7, #95bfa1);
    color: #2d4739;
    font-family: 'Arial', sans-serif;
}

/* ====== Kotak Hijau Judul Utama ====== */
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
}

/* ====== Kotak Pilih Mode & Upload ====== */
.section-box {
    background: linear-gradient(145deg, #7ba883, #547a64);
    border-radius: 20px;
    border: 2px solid #c9e7c0;
    padding: 25px;
    color: #d6edc7;
    box-shadow: 4px 4px 8px rgba(0,0,0,0.25);
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

# ====== Judul Utama ======
st.markdown('<div class="main-title">🧠 Deteksi dan Klasifikasi Gambar</div>', unsafe_allow_html=True)

# ====== Layout ======
col1, col2 = st.columns([1, 2])

# ---- Kolom Kiri: Pilih Mode ----
with col1:
    st.markdown('<div class="section-title">⚙️ Pilih Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Kolom Kanan: Upload & Hasil ----
with col2:
    st.markdown('<div class="section-title">📤 Upload & Hasil Deteksi / Klasifikasi</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Seret atau pilih gambar di sini 👇", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        if mode == "Deteksi Objek (YOLO)":
            # ==== DETEKSI OBJEK DENGAN BOUNDING BOX MANUAL ====
            img_array = np.array(img)
            results = yolo_model(img_array)
            img_with_boxes = img_array.copy()

            # Ambil nama kelas dari model YOLO
            class_names = yolo_model.names

            for box in results[0].boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label_index = int(box.cls[0])
                label_name = class_names[label_index]

                # Gambar kotak hijau
                cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # Tambah label + confidence
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

            # ✅ Opsional: kalau kamu punya daftar nama kelas klasifikasi
            # ganti bagian bawah ini:
            class_labels = ["Kelas 1", "Kelas 2", "Kelas 3", "Kelas 4", "Kelas 5"]  # sesuaikan
            class_name = class_labels[class_index] if class_index < len(class_labels) else str(class_index)

            st.markdown(
                f'<div class="detect-result">📊 <b>Hasil Prediksi:</b> {class_name}<br>🎯 <b>Akurasi:</b> {accuracy:.2f}%</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("Silakan unggah gambar terlebih dahulu di atas.")
    st.markdown('</div>', unsafe_allow_html=True)
