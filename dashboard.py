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
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #a3cbb8, #6a9080);
    color: #cadfc7;
    font-family: 'Arial', sans-serif;
    padding: 2rem 3rem;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }

.welcome-box {
    background: linear-gradient(145deg, #57876a, #9dbcae);
    border-radius: 12px;
    padding: 15px 20px;
    margin-bottom: 25px;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
    border: 1.5px solid #c9e7c0;
}
.main-box {
    background: linear-gradient(145deg, #8daaa5, #618472);
    border-radius: 15px;
    padding: 40px 25px;
    margin-bottom: 25px;
    font-weight: bold;
    font-size: 28px;
    text-align: center;
    border: 2px solid #c9e7c0;
    box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
}
.main-title {
    background: linear-gradient(145deg, #6b9474, #547a64);
    border: 3px solid #c9e7c0;
    border-radius: 20px;
    color: #eaf4e2;
    text-align: center;
    padding: 20px;
    font-size: 28px;
    font-weight: bold;
    margin: 20px auto 25px auto;
    box-shadow: 4px 4px 8px rgba(0,0,0,0.25);
}
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
.mode-explanation {
    background: #6f9b7c;
    border: 2px solid #c9e7c0;
    border-radius: 12px;
    padding: 15px;
    margin-top: 15px;
    color: #f1f8ee;
    font-size: 15px;
    text-align: justify;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
}
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
.return-button {
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# ====== Session state ======
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'


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


# ====== Halaman Utama ======
def halaman_main():
    st.markdown('<div class="main-title">🧠 Deteksi dan Klasifikasi Gambar</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])

    # Kolom kiri = Pilih Mode
    with col1:
        st.markdown('<div class="section-title">⚙️ Pilih Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

        # Penjelasan dinamis
        if mode == "Deteksi Objek (YOLO)":
            st.markdown("""
            <div class="mode-explanation">
            Mode ini menggunakan model <b>YOLO (You Only Look Once)</b> untuk mendeteksi berbagai objek dalam gambar.
            Setiap objek akan diberi <b>bounding box hijau</b> lengkap dengan label dan tingkat kepercayaannya.
            Cocok digunakan untuk menganalisis gambar yang memiliki lebih dari satu objek.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="mode-explanation">
            Mode ini menjalankan proses <b>klasifikasi gambar</b> menggunakan model CNN (Convolutional Neural Network).
            Gambar akan diubah menjadi ukuran 224x224 piksel, lalu diprediksi ke dalam salah satu kelas yang tersedia.
            Mode ini berguna untuk mengenali jenis atau kategori dari satu gambar utama.
            </div>
            """, unsafe_allow_html=True)

    # Kolom kanan = Upload & hasil
    with col2:
        st.markdown('<div class="section-title">📤 Upload & Hasil Deteksi / Klasifikasi</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Seret atau pilih gambar di sini 👇", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            img_array = np.array(img)

            if mode == "Deteksi Objek (YOLO)":
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

                # tampilkan dua gambar berdampingan dan kecil
                colA, colB = st.columns(2)
                with colA:
                    st.image(img, caption="🖼️ Gambar Asli", width=250)
                with colB:
                    st.image(img_with_boxes, caption="📦 Hasil Deteksi", width=250)

                st.markdown('<div class="detect-result">✅ Deteksi objek berhasil dilakukan.</div>', unsafe_allow_html=True)

            elif mode == "Klasifikasi Gambar":
                img_resized = img.resize((224, 224))
                arr = image.img_to_array(img_resized)
                arr = np.expand_dims(arr, axis=0) / 255.0
                prediction = classifier.predict(arr)
                class_index = np.argmax(prediction)
                accuracy = float(np.max(prediction)) * 100

                class_labels = ["Kelas 1", "Kelas 2", "Kelas 3", "Kelas 4", "Kelas 5"]
                class_name = class_labels[class_index] if class_index < len(class_labels) else str(class_index)

                # tampilkan gambar kecil
                st.image(img, caption="🖼️ Gambar Diupload", width=250)
                st.markdown(
                    f'<div class="detect-result">📊 <b>Hasil Prediksi:</b> {class_name}<br>🎯 <b>Akurasi:</b> {accuracy:.2f}%</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Silakan unggah gambar terlebih dahulu di atas.")

    st.markdown('<div class="return-button"></div>', unsafe_allow_html=True)
    if st.button("Kembali ke Halaman Awal"):
        st.session_state['page'] = 'home'


# ====== Routing ======
if st.session_state['page'] == 'home':
    halaman_awal()
elif st.session_state['page'] == 'main':
    halaman_main()
