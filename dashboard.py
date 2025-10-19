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
    margin: 0;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }

.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
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
    width: 100%;
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

.explain-box {
    background: #7ba883;
    border: 2px solid #c9e7c0;
    border-radius: 10px;
    padding: 15px;
    color: #eaf4e2;
    margin-top: 10px;
    font-weight: 500;
    text-align: justify;
}
</style>
""", unsafe_allow_html=True)

# ====== Session state ======
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# ====== HALAMAN AWAL BARU ======
def halaman_awal():
    st.set_page_config(page_title="Dashboard Akbar Dzikri", layout="wide")

    st.markdown("""
        <div style='text-align:center; background-color:#98b8ac; padding:15px; 
                    border-radius:15px; box-shadow:0px 4px 6px rgba(0,0,0,0.2);'>
            🌸 <b>SELAMAT DATANG DI DASHBOARD MUHAMMAD AKBAR DZIKRI</b> 🌸
        </div>
    """, unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)

    # ================================
    # Kolom 1 – Klasifikasi Gambar
    # ================================
    with col1:
        st.markdown("""
            <div style='background-color:#a5c1b2; padding:35px; border-radius:20px;
                        box-shadow:2px 4px 10px rgba(0,0,0,0.2); text-align:center;'>
                <h2 style='color:#f7fff7;'>🌼 KLASIFIKASI GAMBAR 🌼</h2>
                <p style='color:#f1f1f1; font-size:16px; text-align:justify; line-height:1.6;'>
                    Fitur ini dirancang untuk mengidentifikasi jenis bunga berdasarkan gambar 
                    yang kamu unggah. Model yang digunakan adalah <b>Convolutional Neural Network (CNN)</b>,
                    yang telah dilatih menggunakan ratusan gambar bunga dari berbagai kategori 
                    seperti <b>Daisy, Dandelion, Rose, Sunflower,</b> dan <b>Tulip</b>.
                    <br><br>
                    Saat kamu mengunggah gambar, sistem akan melakukan proses ekstraksi fitur,
                    menganalisis pola visual seperti bentuk kelopak, warna dominan, dan tekstur permukaan bunga.
                    Setelah itu, hasil prediksi akan ditampilkan.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # ================================
    # Kolom 2 – Deteksi Objek
    # ================================
    with col2:
        st.markdown("""
            <div style='background-color:#a5c1b2; padding:35px; border-radius:20px;
                        box-shadow:2px 4px 10px rgba(0,0,0,0.2); text-align:center;'>
                <h2 style='color:#f7fff7;'>🌻 DETEKSI OBJEK 🌻</h2>
                <p style='color:#f1f1f1; font-size:16px; text-align:justify; line-height:1.6;'>
                    Fitur ini menggunakan model <b>YOLO (You Only Look Once)</b> yang berfungsi untuk mendeteksi 
                    keberadaan bunga secara otomatis dalam suatu gambar. 
                    Model akan mencari area-area yang berpotensi mengandung objek bunga dan 
                    menampilkan hasil deteksi dalam bentuk <b>kotak hijau</b> di sekitar objek tersebut.
                    <br><br>
                    Dengan fitur ini, pengguna tidak perlu lagi menandai objek secara manual — 
                    cukup unggah gambar, dan sistem akan secara otomatis melakukan deteksi serta klasifikasi secara bersamaan.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    if st.button("HALAMAN BERIKUTNYA →"):
        st.session_state['page'] = 'main'

# ====== HALAMAN UTAMA (tidak diubah) ======
def halaman_main():
    st.markdown('<div class="main-title">🧠 Deteksi dan Klasifikasi Gambar</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">⚙️ Pilih Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
        if mode == "Deteksi Objek (YOLO)":
            st.markdown("""
            <div class="explain-box">
                <b>Mode Deteksi Objek (YOLO):</b><br>
                Sistem akan mendeteksi setiap objek di gambar, memberi label, dan klasifikasi tambahan.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="explain-box">
                <b>Mode Klasifikasi Gambar:</b><br>
                Sistem akan menentukan kelas keseluruhan gambar menggunakan model klasifikasi.
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">📤 Upload Gambar Sesuai Mode</div>', unsafe_allow_html=True)

        # ====== MODE YOLO ======
        if mode == "Deteksi Objek (YOLO)":
            uploaded_yolo = st.file_uploader("Unggah gambar untuk Deteksi Objek 👇", type=["jpg", "jpeg", "png"], key="yolo")
            if uploaded_yolo is not None:
                img = Image.open(uploaded_yolo)
                img_array = np.array(img)
                results = yolo_model(img_array)
                img_with_boxes = img_array.copy()
                class_names = yolo_model.names
                detected_objects = []

                for box in results[0].boxes:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    label_index = int(box.cls[0])
                    yolo_label = class_names[label_index]
                    cropped_obj = img_array[ymin:ymax, xmin:xmax]
                    cropped_pil = Image.fromarray(cropped_obj).resize((224, 224))
                    cropped_arr = image.img_to_array(cropped_pil)
                    cropped_arr = np.expand_dims(cropped_arr, axis=0) / 255.0
                    pred = classifier.predict(cropped_arr)
                    idx = np.argmax(pred)
                    acc = float(np.max(pred)) * 100
                    labels = ["Kelas 1 (Daisy)", "Kelas 2 (Dandelion)", "Kelas 3 (Rose)", "Kelas 4 (Sunflower)", "Kelas 5 (Tulip)"]
                    class_name = labels[idx] if idx < len(labels) else str(idx)
                    cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                    cv2.putText(img_with_boxes, f"{class_name} ({acc:.1f}%)", (xmin, max(ymin - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
                    detected_objects.append((yolo_label, class_name, acc))

                col_yolo1, col_yolo2 = st.columns([1, 1], gap="large")
                with col_yolo1:
                    st.image(img, caption="🖼️ Gambar Asli", use_container_width=True)
                with col_yolo2:
                    st.image(img_with_boxes, caption="📦 Hasil Deteksi & Klasifikasi", use_container_width=True)

                st.markdown('<div class="detect-result">✅ <b>Hasil Deteksi dan Klasifikasi:</b></div>', unsafe_allow_html=True)
                for i, (det, cls, acc) in enumerate(detected_objects):
                    st.markdown(f"""
                    <div class="detect-result">
                        🌼 <b>Objek {i+1}</b><br>
                        🔍 <b>Deteksi:</b> {det}<br>
                        📊 <b>Klasifikasi:</b> {cls}<br>
                        🎯 <b>Akurasi:</b> {acc:.2f}%
                    </div>
                    """, unsafe_allow_html=True)

        # ====== MODE KLASIFIKASI ======
        elif mode == "Klasifikasi Gambar":
            uploaded_class = st.file_uploader("Unggah gambar untuk Klasifikasi 👇", type=["jpg", "jpeg", "png"], key="classify")
            if uploaded_class is not None:
                img = Image.open(uploaded_class)
                img_resized = img.resize((224, 224))
                arr = image.img_to_array(img_resized)
                arr = np.expand_dims(arr, axis=0) / 255.0
                pred = classifier.predict(arr)
                idx = np.argmax(pred)
                acc = float(np.max(pred)) * 100
                labels = ["Kelas 1 (Daisy)", "Kelas 2 (Dandelion)", "Kelas 3 (Rose)", "Kelas 4 (Sunflower)", "Kelas 5 (Tulip)"]
                class_name = labels[idx] if idx < len(labels) else str(idx)
                st.image(img, caption="🖼️ Gambar Diupload", width=300)
                st.markdown(f"""
                <div class="detect-result">
                    📊 <b>Hasil Prediksi:</b> {class_name}<br>
                    🎯 <b>Akurasi:</b> {acc:.2f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Silakan unggah gambar untuk klasifikasi di atas.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Kembali ke Halaman Awal"):
        st.session_state['page'] = 'home'


# ====== Routing Halaman ======
if st.session_state['page'] == 'home':
    halaman_awal()
elif st.session_state['page'] == 'main':
    halaman_main()
