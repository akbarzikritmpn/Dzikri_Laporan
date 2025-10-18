import streamlit as st

# ====== CSS Halaman Awal ======
st.markdown("""
<style>
/* Background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #a3cbb8, #6a9080);
    color: #cadfc7;
    font-family: 'Arial', sans-serif;
    padding: 2rem 3rem;
    margin: 0;
}

/* Kotak Selamat Datang */
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

/* Kotak Judul Utama */
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

/* Tombol dalam kotak */
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

/* Buat tombol seperti tombol Streamlit */
.button-box button {
    all: unset;
    cursor: pointer;
    font-weight: bold;
    font-size: 14px;
    color: inherit;
    width: 100%;
    display: block;
    user-select: none;
}
</style>
""", unsafe_allow_html=True)

# ====== Konten Halaman Awal ======
st.markdown('<div class="welcome-box">SELAMAT DATANG DI DASHBOARD MUHAMMAD AKBAR DZIKRI</div>', unsafe_allow_html=True)

st.markdown("""
<div class="main-box">
KLASIFIKASI GAMBAR <br> & <br> OBJEK DETECTION
</div>
""", unsafe_allow_html=True)

# Tombol halaman berikutnya dalam kotak
clicked = st.markdown("""
<div class="button-box">
    <button id="next-btn">HALAMAN BERIKUTNYA</button>
</div>
""", unsafe_allow_html=True)

# Karena button HTML custom tidak langsung bisa dipakai untuk interaksi,
# kita buat tombol Streamlit transparan di bawahnya untuk navigasi:

if st.button("HALAMAN BERIKUTNYA"):
    st.session_state['page'] = "second"

# Navigasi sederhana berdasarkan session_state
if 'page' not in st.session_state:
    st.session_state['page'] = "first"

if st.session_state['page'] == "first":
    pass  # Tetap di halaman awal

elif st.session_state['page'] == "second":
    # Halaman kedua, contoh sederhana
    st.write("Ini halaman kedua, kamu bisa masukkan syntax halaman lain di sini.")
    if st.button("Kembali ke Halaman Awal"):
        st.session_state['page'] = "first"
