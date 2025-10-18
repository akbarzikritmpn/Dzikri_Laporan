import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====== CSS Custom kamu ======
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
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="section-box"><div class="section-title">📅 Contract Start</div>20-Jan-16</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="section-box"><div class="section-title">📅 Contract Finish</div>09-Nov-17</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="section-box"><div class="section-title">📅 Forecast Finish</div>23-Dec-17</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
with col4:
    st.markdown('<div class="section-box"><div class="section-title">📉 Delay/Ahead (Days)</div>-44</div>', unsafe_allow_html=True)
with col5:
    st.markdown('<div class="section-box"><div class="section-title">📊 Variance %</div>-1.12%</div>', unsafe_allow_html=True)
with col6:
    st.markdown('<div class="section-box"><div class="section-title">⭐ CPI</div>0.89</div>', unsafe_allow_html=True)

# ====== Progress Curve (Cost) dengan Matplotlib ======
st.markdown('<div class="section-title">📈 Progress Curve - Cost</div>', unsafe_allow_html=True)

dates = pd.date_range(start="2016-01-01", periods=12, freq='M')
planned = np.linspace(0, 100, 12)
actual = planned + np.random.normal(0, 5, 12)

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(dates, planned, label='Planned', color='#2d4739', linewidth=2)
ax1.plot(dates, actual, label='Actual', color='#a2c3a4', linestyle='--', linewidth=2)
ax1.set_title("Progress Planned vs Actual", fontsize=14, color="#2d4739")
ax1.set_ylabel("Progress (%)")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# ====== Bar Chart: Planned vs Actual (Activity Wise) ======
st.markdown('<div class="section-title">📋 Activity-wise Progress (%)</div>', unsafe_allow_html=True)

activities = ['Structure', 'MEP', 'Finishes', 'Testing']
planned_pct = [90, 75, 60, 40]
actual_pct = [85, 70, 55, 35]
x = np.arange(len(activities))

fig2, ax2 = plt.subplots(figsize=(6, 4))
bar_width = 0.35
ax2.bar(x - bar_width/2, planned_pct, width=bar_width, label='Planned', color='#2d4739')
ax2.bar(x + bar_width/2, actual_pct, width=bar_width, label='Actual', color='#a2c3a4')
ax2.set_xticks(x)
ax2.set_xticklabels(activities)
ax2.set_ylabel('Progress (%)')
ax2.set_title("Planned vs Actual Progress")
ax2.legend()
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
st.pyplot(fig2)
