import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2

# ====== Load Model ======
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/Muhammad Akbar Dzikri_Laporan 4.pt")
        classifier = tf.keras.models.load_model("model/Zikri_Laporan2.h5")
        return yolo_model, classifier
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

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

# ====== Session State ======
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# ====== HALAMAN AWAL ======
def halaman_awal():
    st.set_page_config(page_title="Dashboard Akbar Dzikri", layout="wide")
    st.markdown("""
        <div class="main-title">
            🌸 <b>SELAMAT DATANG DI DASHBOARD MUHAMMAD AKBAR DZIKRI</b> 🌸
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="section-title">
            <b>Penjelasan Awal Sebelum Menuju Dashboard</b>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="section-title">🌼 KLASIFIKASI GAMBAR 🌼</div>
            <div class="explain-box">
                Fitur ini dirancang untuk mengidentifikasi jenis bunga berdasarkan gambar 
                yang kamu unggah. Model menggunakan <b>CNN</b> dan dapat mengenali 
                <b>Daisy, Dandelion, Rose, Sunflower,</b> serta <b>Tulip</b>.
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="section-title">🌻 DETEKSI OBJEK 🌻</div>
            <div class="explain-box">
                Fitur ini memakai model <b>YOLO</b> untuk mendeteksi bunga secara otomatis 
                dan menampilkan kotak hijau di sekitar objek terdeteksi.
            </div>
        """, unsafe_allow_html=True)

    if st.button("HALAMAN BERIKUTNYA →"):
        st.session_state['page'] = 'main'

# ====== HALAMAN UTAMA ======
def halaman_main():
    st.markdown('<div class="main-title">🧠 Deteksi dan Klasifikasi Gambar</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        mode = st.radio("Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

    with col2:
        if mode == "Deteksi Objek (YOLO)":
            uploaded_yolo = st.file_uploader("Unggah gambar untuk Deteksi Objek 👇", type=["jpg", "jpeg", "png"], key="yolo")

            if uploaded_yolo:
                try:
                    img = Image.open(uploaded_yolo)
                    img_array = np.array(img)
                    results = yolo_model(img_array)

                    if len(results[0].boxes) == 0:
                        st.warning("🚫 Tidak ada objek bunga terdeteksi pada gambar ini.")
                        return

                    img_with_boxes = img_array.copy()
                    class_names = yolo_model.names
                    detected_objects = []

                    for box in results[0].boxes:
                        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                        label_index = int(box.cls[0])
                        yolo_label = class_names[label_index]
                        cropped_obj = img_array[ymin:ymax, xmin:xmax]
                        if cropped_obj.size == 0:
                            continue

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

                    st.image(img_with_boxes, caption="📦 Hasil Deteksi & Klasifikasi", use_container_width=True)
                    for i, (det, cls, acc) in enumerate(detected_objects):
                        st.markdown(f"""
                        <div class="detect-result">
                            🌼 <b>Objek {i+1}</b><br>
                            🔍 <b>Deteksi:</b> {det}<br>
                            📊 <b>Klasifikasi:</b> {cls}<br>
                            🎯 <b>Akurasi:</b> {acc:.2f}%
                        </div>
                        """, unsafe_allow_html=True)

                except UnidentifiedImageError:
                    st.error("⚠️ File yang diunggah bukan gambar yang valid.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

        elif mode == "Klasifikasi Gambar":
            uploaded_class = st.file_uploader("Unggah gambar untuk Klasifikasi 👇", type=["jpg", "jpeg", "png"], key="classify")
            if uploaded_class:
                try:
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

                except UnidentifiedImageError:
                    st.error("⚠️ File yang diunggah bukan gambar yang valid.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
            else:
                st.info("Silakan unggah gambar untuk klasifikasi di atas.")

    if st.button("⬅️ Kembali ke Halaman Awal"):
        st.session_state['page'] = 'home'


# ====== Routing Halaman ======
if st.session_state['page'] == 'home':
    halaman_awal()
else:
    halaman_main()
