import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ====== CSS Kustom kamu (reused from YOLO app) ======
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #d6edc7, #95bfa1);
    color: #2d4739;
    font-family: 'Arial', sans-serif;
}

.main-title {
    background: linear-gradient(145deg, #6b9474, #547a64);
    border: 3px solid #c9e7c0;
    border-radius: 20px;
    color: #eaf4e2;
    text-align: center;
    padding: 20px;
    font-size: 30px;
    font-weight: bold;
    margin-bottom: 30px;
    box-shadow: 4px 4px 8px rgba(0,0,0,0.25);
}

.section-box {
    background: linear-gradient(145deg, #7ba883, #547a64);
    border-radius: 20px;
    border: 2px solid #c9e7c0;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 4px 4px 8px rgba(0,0,0,0.25);
    color: #eaf4e2;
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
</style>
""", unsafe_allow_html=True)

# ====== Judul Utama ======
st.markdown('<div class="main-title">📊 Full Project Analytics Dashboard</div>', unsafe_allow_html=True)

# ====== Informasi Proyek ======
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="section-box"><div class="section-title">📅 Contract Start</div>20-Jan-16</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-box"><div class="section-title">📅 Contract Finish</div>09-Nov-17</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="section-box"><div class="section-title">📅 Forecast Finish</div>23-Dec-17</div>', unsafe_allow_html=True)

# ====== Delay & Variance Info ======
col4, col5, col6 = st.columns(3)
with col4:
    st.markdown('<div class="section-box"><div class="section-title">📉 Delay/Ahead (Days)</div>-44</div>', unsafe_allow_html=True)
with col5:
    st.markdown('<div class="section-box"><div class="section-title">📊 Variance %</div>-1.12%</div>', unsafe_allow_html=True)
with col6:
    st.markdown('<div class="section-box"><div class="section-title">⭐ CPI</div>0.89</div>', unsafe_allow_html=True)

# ====== Grafik Dummy: Progress Curve ======
st.markdown('<div class="section-title">📈 Progress Curve - Cost</div>', unsafe_allow_html=True)

# Simulasi data
dates = pd.date_range(start="2016-01-01", periods=12, freq='M')
planned = np.linspace(0, 100, 12)
actual = planned + np.random.normal(0, 5, 12)

df_progress = pd.DataFrame({"Date": dates, "Planned": planned, "Actual": actual})

fig = px.line(df_progress, x="Date", y=["Planned", "Actual"],
              labels={"value": "Progress (%)", "Date": "Date"},
              title="Progress Planned vs Actual")

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(255,255,255,0)',
                  title_font=dict(size=20, color="#2d4739"),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

st.plotly_chart(fig, use_container_width=True)

# ====== Planned vs Actual % (Activity Wise) Placeholder ======
st.markdown('<div class="section-title">📋 Planned vs Actual (%) Activity Wise</div>', unsafe_allow_html=True)

# Dummy data bar chart
activity_data = pd.DataFrame({
    "Activity": ["Structure", "MEP", "Finishes", "Testing"],
    "Planned %": [90, 75, 60, 40],
    "Actual %": [85, 70, 55, 35]
})

fig2 = px.bar(activity_data, x="Activity", y=["Planned %", "Actual %"],
              barmode="group", title="Activity Progress Comparison")

fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(255,255,255,0)',
                   title_font=dict(size=20, color="#2d4739"),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

st.plotly_chart(fig2, use_container_width=True)
