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
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }

.welcome-box {
    background: linear-gradient(145deg, #57876a, #9dbcae);
    border-radius: 12px;
    padding: 15px 20px;
    margin-bottom: 25px;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
    border: 1.5px solid #c9e7c0;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
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
    line-height: 1.3;
}
.button-box {
    background: linear-gradient(145deg, #7e9c7d, #55775b);
    border-radius: 12px;
    padding: 10px 25px;
    width: 160px;
    margin: 0 auto;
    text-align: center;
    font-weight: bold;
    color: #cadfc7;
    cursor: pointer;
    border: 1.5px solid #c9e7c0;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    user-select: none;
    transition: background 0.3s ease;
}
.button-box:hover {
    background: linear-gradient(145deg, #55775b, #7e9c7d);
}
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
.section-box {
    background: linear-gradient(145deg, #7ba883, #547a64);
    border-radius: 20px;
    border: 2px solid #c9e7c0;
    padding: 25px;
    color: #d6edc7;
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
div[data-testid="stFileUploader"] {
    background: #7ba883;
    border: 2px dashed #c9e7c0;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    color: #f0f8ec !important;
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

    with col1:
        st.markdown('<div class="section-title">⚙️ Pilih Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
        st.markdown('</div>', unsafe_allow_html=True)

        if mode == "Deteksi Objek (YOLO)":
            explanation = """
            <div class="explain-box">
            <b>Mode Deteksi Objek (YOLO):</b><br>
            Sistem mendeteksi setiap objek dalam gambar, memberikan label dan tingkat kepercayaan.
            Selain itu, setiap objek yang terdeteksi juga diklasifikasikan menggunakan model klasifikasi tambahan.
            </div>
            """
        else:
            explanation = """
            <div class="explain-box">
            <b>Mode Klasifikasi Gambar:</b><br>
            Mode ini mengklasifikasikan keseluruhan gambar ke dalam satu kelas tertentu berdasarkan model yang telah dilatih.
            </div>
            """
        st.markdown(explanation, unsafe_allow_html=True)

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

                detected_objects = []

                for box in results[0].boxes:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    label_index = int(box.cls[0])
                    yolo_label = model.names

                    # --- Crop objek untuk klasifikasi tambahan ---
                    cropped_obj = img_array[ymin:ymax, xmin:xmax]
                    cropped_obj_pil = Image.fromarray(cropped_obj).resize((224, 224))
                    cropped_obj_array = image.img_to_array(cropped_obj_pil)
                    cropped_obj_array = np.expand_dims(cropped_obj_array, axis=0) / 255.0

                    # Prediksi kelas dari model klasifikasi
                    class_pred = classifier.predict(cropped_obj_array)
                    class_index = np.argmax(class_pred)
                    accuracy = float(np.max(class_pred)) * 100
                    class_labels = ["Kelas 1", "Kelas 2", "Kelas 3", "Kelas 4", "Kelas 5"]
                    class_name = class_labels[class_index] if class_index < len(class_labels) else str(class_index)

                    # Gambar bounding box
                    cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"{class_name} ({accuracy:.1f}%)",
                                (xmin, max(ymin - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    detected_objects.append((yolo_label, class_name, accuracy))

                colA, colB = st.columns(2)
                with colA:
                    st.image(img, caption="🖼️ Gambar Asli", use_container_width=False, width=300)
                with colB:
                    st.image(img_with_boxes, caption="📦 Deteksi & Klasifikasi Multi-Objek", use_container_width=False, width=300)

                # Tampilkan hasil deteksi dan klasifikasi per objek
                st.markdown('<div class="detect-result">✅ Semua objek berhasil dideteksi dan diklasifikasikan:</div>', unsafe_allow_html=True)
                for i, (det_label, cls_label, acc) in enumerate(detected_objects):
                    st.markdown(f"- **Objek {i+1}:** Deteksi YOLO = `{det_label}`, Klasifikasi = `{cls_label}`, Akurasi = `{acc:.2f}%`")

            elif mode == "Klasifikasi Gambar":
                img_resized = img.resize((224, 224))
                img_array2 = image.img_to_array(img_resized)
                img_array2 = np.expand_dims(img_array2, axis=0)
                img_array2 = img_array2 / 255.0

                prediction = classifier.predict(img_array2)
                class_index = np.argmax(prediction)
                accuracy = float(np.max(prediction)) * 100

                class_labels = ["Kelas 1", "Kelas 2", "Kelas 3", "Kelas 4", "Kelas 5"]
                class_name = class_labels[class_index] if class_index < len(class_labels) else str(class_index)

                st.image(img, caption="🖼️ Gambar Diupload", use_container_width=False, width=300)
                st.markdown(
                    f'<div class="detect-result">📊 <b>Hasil Prediksi:</b> {class_name}<br>🎯 <b>Akurasi:</b> {accuracy:.2f}%</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Silakan unggah gambar terlebih dahulu di atas.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Kembali ke Halaman Awal"):
        st.session_state['page'] = 'home'

# ====== Routing Halaman ======
if st.session_state['page'] == 'home':
    halaman_awal()
elif st.session_state['page'] == 'main':
    halaman_main()
