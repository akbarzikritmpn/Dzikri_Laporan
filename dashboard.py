import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go

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

# ====== Session State ======
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
    st.markdown('<div class="main-box">🧠 Deteksi dan Klasifikasi Gambar</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-title">⚙️ Pilih Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

        if mode == "Deteksi Objek (YOLO)":
            explanation = """
            <div class="explain-box">
            <b>Mode Deteksi Objek (YOLO):</b><br>
            Sistem akan mendeteksi dan memberi kotak (bounding box) pada objek di gambar 
            beserta label dan tingkat kepercayaannya.
            </div>
            """
        else:
            explanation = """
            <div class="explain-box">
            <b>Mode Klasifikasi Gambar:</b><br>
            Sistem akan mengklasifikasikan gambar ke dalam kategori tertentu 
            dan menampilkan grafik interaktif probabilitas tiap kelas.
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

                for box in results[0].boxes:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    label_index = int(box.cls[0])
                    label_name = class_names[label_index]
                    cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"{label_name} {confidence:.2f}",
                                (xmin, max(ymin - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)

                colA, colB = st.columns(2)
                with colA:
                    st.image(img, caption="🖼️ Gambar Asli", width=300)
                with colB:
                    st.image(img_with_boxes, caption="📦 Hasil Deteksi", width=300)

                st.markdown('<div class="detect-result">✅ Deteksi objek berhasil dilakukan.</div>', unsafe_allow_html=True)

            elif mode == "Klasifikasi Gambar":
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = classifier.predict(img_array)[0]
                class_index = np.argmax(prediction)
                accuracy = float(np.max(prediction)) * 100

                class_labels = ["Kelas 1", "Kelas 2", "Kelas 3", "Kelas 4", "Kelas 5"]
                class_name = class_labels[class_index] if class_index < len(class_labels) else str(class_index)

                colG1, colG2 = st.columns([1, 1])
                with colG1:
                    st.image(img, caption="🖼️ Gambar Diupload", width=300)

                with colG2:
                    fig = go.Figure(
                        data=[go.Bar(
                            x=class_labels,
                            y=prediction * 100,
                            text=[f"{p*100:.2f}%" for p in prediction],
                            textposition='outside',
                            marker=dict(color=['#a8d5ba', '#91c8a8', '#7bb895', '#6aa784', '#5a9472'])
                        )]
                    )
                    fig.update_layout(
                        title="📊 Visualisasi Probabilitas Tiap Kelas",
                        yaxis_title="Persentase (%)",
                        xaxis_title="Kelas",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#eaf4e2", size=13),
                        margin=dict(t=60, b=30, l=30, r=30),
                        transition_duration=700
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    f'<div class="detect-result">📊 <b>Prediksi:</b> {class_name}<br>🎯 <b>Akurasi:</b> {accuracy:.2f}%</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Silakan unggah gambar terlebih dahulu di atas.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Kembali ke Halaman Awal"):
        st.session_state['page'] = 'home'


# ====== Routing Halaman ======
if st.session_state['page'] == 'home':
    halaman_awal()
elif st.session_state['page'] == 'main':
    halaman_main()
