import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import plotly.express as px  # untuk pie chart

#Load Model
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Muhammad Akbar Dzikri_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Zikri_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

#CSS dengan Animasi
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

/* ===== Animasi Muncul ===== */
@keyframes fadeSlideIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* ===== Efek umum untuk semua kotak ===== */
.main-title, .section-title, .detect-result, .explain-box {
    transition: all 0.3s ease-in-out;
    transform: scale(1);
    animation: fadeSlideIn 0.8s ease forwards;
}

/* Efek hover: sedikit membesar dan bayangan muncul */
.main-title:hover, .section-title:hover, .detect-result:hover, .explain-box:hover {
    transform: scale(1.03);
    box-shadow: 6px 6px 15px rgba(0,0,0,0.25);
}

/* Efek ketika ditekan (klik) — sedikit mengecil */
.main-title:active, .section-title:active, .detect-result:active, .explain-box:active {
    transform: scale(0.97);
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
}

/* ===== Style Asli ===== */
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

#Session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

#HALAMAN AWAL
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

    st.write("")
    col1, col2 = st.columns(2)

    # Kolom 1 – Klasifikasi
    with col1:
        st.markdown("""
            <div class="section-title">🌼 KLASIFIKASI GAMBAR 🌼</div>
            <div class="explain-box">
                Fitur ini dirancang untuk mengidentifikasi jenis bunga berdasarkan gambar 
                yang kamu unggah. Model yang digunakan adalah <b>Convolutional Neural Network (CNN)</b>,
                yang telah dilatih menggunakan ratusan gambar bunga dari berbagai kategori 
                seperti <b>Daisy, Dandelion, Rose, Sunflower,</b> dan <b>Tulip</b>.
                <br><br>
                Saat kamu mengunggah gambar, sistem akan melakukan proses ekstraksi fitur,
                menganalisis pola visual seperti bentuk kelopak, warna dominan, dan tekstur permukaan bunga.
                Setelah itu, hasil prediksi akan ditampilkan.
            </div>
        """, unsafe_allow_html=True)

    # Kolom 2 – Deteksi
    with col2:
        st.markdown("""
            <div class="section-title">🌻 DETEKSI OBJEK 🌻</div>
            <div class="explain-box">
                Fitur ini menggunakan model <b>YOLO (You Only Look Once)</b> yang berfungsi untuk mendeteksi 
                keberadaan bunga secara otomatis dalam suatu gambar. 
                Model akan mencari area-area yang berpotensi mengandung objek bunga dan 
                menampilkan hasil deteksi dalam bentuk <b>kotak hijau</b> di sekitar objek tersebut.
                <br><br>
                Dengan fitur ini, pengguna tidak perlu lagi menandai objek secara manual -
                cukup unggah gambar, dan sistem akan secara otomatis melakukan deteksi serta klasifikasi secara bersamaan.
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    if st.button("HALAMAN BERIKUTNYA →"):
        st.session_state['page'] = 'main'


#HALAMAN UTAMA
def halaman_main():
    st.markdown('<div class="main-title">🔎 Deteksi dan Klasifikasi Gambar 🔍 </div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">⚙️ Pilih Mode yang digunakan</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-title">📤 Upload Gambar</div>', unsafe_allow_html=True)

        uploaded_img = None
        detected_objects = []

        #Deteksi objek
        if mode == "Deteksi Objek (YOLO)":
            uploaded_img = st.file_uploader("Unggah gambar untuk Deteksi Objek 👇", type=["jpg", "jpeg", "png"], key="yolo")
            if uploaded_img is not None:
                try:
                    img = Image.open(uploaded_img).convert("RGB")
                    img_array = np.array(img)

                    if img_array.size == 0:
                        st.error("❌ Gambar tidak valid atau kosong.")
                    else:
                        results = yolo_model(img_array)
                        img_with_boxes = img_array.copy()
                        class_names = yolo_model.names

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
                            labels = ["Kelas 1 (Daisy)", "Kelas 2 (Dandelion)", "Kelas 3 (Rose)",
                                      "Kelas 4 (Sunflower)", "Kelas 5 (Tulip)"]
                            class_name = labels[idx] if idx < len(labels) else str(idx)
                            cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                            cv2.putText(img_with_boxes, f"{class_name} ({acc:.1f}%)",
                                        (xmin, max(ymin - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
                            detected_objects.append((yolo_label, class_name, acc))

                        col_yolo1, col_yolo2 = st.columns([1,1], gap="large")
                        with col_yolo1:
                            st.image(img, caption="🖼️ Gambar Asli", use_container_width=True)
                        with col_yolo2:
                            st.image(img_with_boxes, caption="📦 Hasil Deteksi & Klasifikasi", use_container_width=True)

                        if len(detected_objects) == 0:
                            st.warning("⚠️ Tidak ada objek terdeteksi.")
                        else:
                            st.markdown('<div class="detect-result">✅ <b>Hasil Deteksi dan Klasifikasi:</b></div>', unsafe_allow_html=True)
                            for i, (det, cls, acc) in enumerate(detected_objects):
                                st.markdown(f"""
                                <div class="detect-result">
                                    🌼 <b>Objek {i+1}</b><br>
                                    🔍 <b>Deteksi:</b> {det}<br>
                                    📊 <b>Klasifikasi:</b> {cls}<br>
                                    🎯 <b>Akurasi:</b> {acc:.2f}%</div>
                                """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Kesalahan saat memproses gambar: {str(e)}")

        #Klasifikasi
        elif mode == "Klasifikasi Gambar":
            uploaded_img = st.file_uploader("Unggah gambar untuk Klasifikasi 👇", type=["jpg", "jpeg", "png"], key="classify")
            if uploaded_img is not None:
                try:
                    img = Image.open(uploaded_img).convert("RGB")
                    img_resized = img.resize((224,224))
                    arr = image.img_to_array(img_resized)
                    arr = np.expand_dims(arr, axis=0)/255.0
                    pred = classifier.predict(arr)
                    idx = np.argmax(pred)
                    acc = float(np.max(pred)) * 100
                    labels = ["Kelas 1 (Daisy)", "Kelas 2 (Dandelion)", "Kelas 3 (Rose)",
                              "Kelas 4 (Sunflower)", "Kelas 5 (Tulip)"]
                    class_name = labels[idx] if idx < len(labels) else str(idx)
                    st.image(img, caption="🖼️ Gambar Diupload", width=300)
                    st.markdown(f"""
                    <div class="detect-result">
                        📊 <b>Hasil Prediksi:</b> {class_name}<br>
                        🎯 <b>Akurasi:</b> {acc:.2f}%</div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Gagal memproses gambar: {str(e)}")

        #Pie chart
        if uploaded_img is not None:
            if mode == "Klasifikasi Gambar":
                acc_value = acc
            elif mode == "Deteksi Objek (YOLO)" and len(detected_objects)>0:
                acc_value = np.mean([obj[2] for obj in detected_objects])
            else:
                acc_value = None

            if acc_value is not None:
                benar = acc_value
                salah = 100 - acc_value
                pie_data = pd.DataFrame({"Hasil":["Benar","Salah"], "Persentase":[benar,salah]})
                fig = px.pie(pie_data, values='Persentase', names='Hasil',
                             color='Hasil', color_discrete_map={'Benar':'green','Salah':'red'},
                             hole=0.3)
                fig.update_traces(textinfo='label+percent+value', marker=dict(line=dict(color='rgba(0,0,0,0)')))
                fig.update_layout(width=250, height=250, margin=dict(l=0,r=0,t=0,b=0),
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

                # Judul kecil hitam di atas pie chart
                st.markdown('<div class="section-title">Donut Chart Persen Akurasi</div>', unsafe_allow_html=True)
                # Pie chart di tengah
                st.markdown('<div style="display:flex; justify-content:center;">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Kembali ke Halaman Awal"):
        st.session_state['page'] = 'home'


#Routing Halaman
if st.session_state['page'] == 'home':
    halaman_awal()
elif st.session_state['page'] == 'main':
    halaman_main()
