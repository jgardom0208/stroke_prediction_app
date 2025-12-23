import streamlit as st
import pickle

st.set_page_config(page_title="IA Detector de Ictus", page_icon="🧠")

@st.cache_resource
def load_model():
    with open('././models/stroke-model.pck', 'rb') as f:
        return pickle.load(f)

dv, model = load_model()


st.title("🧠 Predicción de Riesgo de Ictus")
st.markdown("""
Esta aplicación utiliza un modelo de **Machine Learning (SVM)** optimizado para detectar indicadores de riesgo de infarto cerebral.
""")

st.sidebar.header("Datos del Paciente")
st.sidebar.info("Modifica los valores para ver la probabilidad en tiempo real.")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Edad", 0, 100, 50)
        gender = st.selectbox("Género", ["Male", "Female", "Other"])
        hypertension = st.radio("¿Tiene Hipertensión?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        heart_disease = st.radio("¿Enfermedad Cardíaca?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        ever_married = st.selectbox("¿Casado/a alguna vez?", ["Yes", "No"])

    with col2:
        avg_glucose_level = st.number_input("Nivel de Glucosa (mg/dL)", 50.0, 300.0, 105.0)
        bmi = st.number_input("IMC (BMI)", 10.0, 60.0, 25.0)
        work_type = st.selectbox("Ocupación", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Zona de Residencia", ["Urban", "Rural"])
        smoking_status = st.selectbox("Tabaquismo", ["formerly smoked", "never smoked", "smokes", "Unknown"])


patient_data = {
    "gender": gender, "age": float(age), "hypertension": hypertension,
    "heart_disease": heart_disease, "ever_married": ever_married,
    "work_type": work_type, "residence_type": residence_type,
    "avg_glucose_level": avg_glucose_level, "bmi": bmi,
    "smoking_status": smoking_status
}


if st.button("Analizar Paciente"):

    X = dv.transform([patient_data])

    prob = model.predict_proba(X)[0, 1]
    

    st.divider()
    st.subheader(f"Resultado del Análisis")

    st.progress(prob)
    st.write(f"**Probabilidad de riesgo:** {round(prob * 100, 2)}%")

    if prob >= 0.5:
        st.error("🚨 **ALTO RIESGO**: El perfil clínico coincide con patrones de pacientes que han sufrido ictus.")
    else:
        st.success("✅ **RIESGO BAJO**: No se detectan anomalías críticas según los datos proporcionados.")

    st.warning("**Nota informativa:** Este es un modelo de IA con fines educativos. Consulta siempre a un médico.")