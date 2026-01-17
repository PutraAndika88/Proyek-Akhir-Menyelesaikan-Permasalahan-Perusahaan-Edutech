import streamlit as st
import pandas as pd
import joblib
import sys
from Preprocessing import DataQualityChecker

sys.modules["__main__"].DataQualityChecker = DataQualityChecker

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = "model/dropout_pipeline_tuned.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.markdown(
    """
Aplikasi ini digunakan untuk **mendeteksi risiko dropout mahasiswa sejak dini**
berdasarkan data akademik, demografi, dan kondisi sosial ekonomi.

📌 **Catatan:**  
Hasil prediksi digunakan sebagai *decision support system*  
dan **bukan keputusan akhir akademik**.
"""
)

st.divider()


st.subheader("📋 Masukkan Data Mahasiswa")

with st.form("student_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Usia Saat Pendaftaran", min_value=15, max_value=70, value=20)
        gender = st.selectbox(
            "Jenis Kelamin",
            options=[0, 1],
            format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki"
        )
        scholarship = st.selectbox(
            "Penerima Beasiswa",
            options=[0, 1],
            format_func=lambda x: "Tidak" if x == 0 else "Ya"
        )
        tuition = st.selectbox(
            "Biaya Kuliah Lunas",
            options=[0, 1],
            format_func=lambda x: "Tidak" if x == 0 else "Ya"
        )

    with col2:
        admission_grade = st.slider("Admission Grade", 0.0, 200.0, 120.0)
        prev_grade = st.slider("Nilai Pendidikan Sebelumnya", 0.0, 200.0, 120.0)
        debtor = st.selectbox(
            "Memiliki Tunggakan",
            options=[0, 1],
            format_func=lambda x: "Tidak" if x == 0 else "Ya"
        )
        displaced = st.selectbox(
            "Mahasiswa Pindahan",
            options=[0, 1],
            format_func=lambda x: "Tidak" if x == 0 else "Ya"
        )

    with col3:
        sem1_approved = st.number_input(
            "MK Lulus Semester 1", min_value=0, max_value=20, value=5
        )
        sem2_approved = st.number_input(
            "MK Lulus Semester 2", min_value=0, max_value=20, value=5
        )
        sem1_grade = st.slider("Rata-rata Nilai Semester 1", 0.0, 20.0, 12.0)
        sem2_grade = st.slider("Rata-rata Nilai Semester 2", 0.0, 20.0, 12.0)

    submitted = st.form_submit_button("🔍 Prediksi Risiko")



if submitted:
    input_data = pd.DataFrame([{
        "Marital_status": 1,
        "Application_mode": 1,
        "Application_order": 1,
        "Course": 171,
        "Daytime_evening_attendance": 1,
        "Previous_qualification": 1,
        "Previous_qualification_grade": prev_grade,
        "Nacionality": 1,
        "Mothers_qualification": 1,
        "Fathers_qualification": 1,
        "Mothers_occupation": 1,
        "Fathers_occupation": 1,
        "Admission_grade": admission_grade,
        "Displaced": displaced,
        "Educational_special_needs": 0,
        "Debtor": debtor,
        "Tuition_fees_up_to_date": tuition,
        "Gender": gender,
        "Scholarship_holder": scholarship,
        "Age_at_enrollment": age,
        "International": 0,
        "Curricular_units_1st_sem_credited": 0,
        "Curricular_units_1st_sem_enrolled": 6,
        "Curricular_units_1st_sem_evaluations": 6,
        "Curricular_units_1st_sem_approved": sem1_approved,
        "Curricular_units_1st_sem_grade": sem1_grade,
        "Curricular_units_1st_sem_without_evaluations": 0,
        "Curricular_units_2nd_sem_credited": 0,
        "Curricular_units_2nd_sem_enrolled": 6,
        "Curricular_units_2nd_sem_evaluations": 6,
        "Curricular_units_2nd_sem_approved": sem2_approved,
        "Curricular_units_2nd_sem_grade": sem2_grade,
        "Curricular_units_2nd_sem_without_evaluations": 0,
        "Unemployment_rate": 10.0,
        "Inflation_rate": 1.0,
        "GDP": 1.5
    }])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    label_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
    risk_label = label_map[prediction]

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    if risk_label == "Dropout":
        st.error("⚠️ **Risiko Tinggi Dropout**")
    elif risk_label == "Enrolled":
        st.warning("⚠️ **Mahasiswa Berpotensi Bermasalah**")
    else:
        st.success("✅ **Mahasiswa Berpotensi Lulus**")

    st.markdown("### Probabilitas Prediksi")
    prob_df = pd.DataFrame({
        "Status": ["Dropout", "Enrolled", "Graduate"],
        "Probabilitas": proba
    })

    st.bar_chart(prob_df.set_index("Status"))



st.divider()
st.caption(
    "Model: XGBoost (Tuned) | "
    "Proyek Data Science – Jaya Jaya Institut | "
    "© 2026"
)
