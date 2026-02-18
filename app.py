import streamlit as st
import joblib
import urllib.request
import os
import pandas as pd
import random

@st.cache_resource
def load_model():
    model_path = 'modelo_random_forest.pkl'
    
    # Si el archivo no está en el servidor de Streamlit, lo bajamos del Release de GitHub
    if not os.path.exists(model_path):
        # SUSTITUYE ESTA URL: Haz clic derecho en el archivo en tu 'Release' y dale a 'Copiar dirección de enlace'
        url = "https://github.com/amartinez113tfm/calidadAireMadrid/releases/download/v1.0/modelo_random_forest.pkl"
        
        with st.spinner("Descargando modelo... un momento, por favor."):
            urllib.request.urlretrieve(url, model_path)
            
    return joblib.load(model_path)

# Ahora ya puedes usarlo normalmente
model = load_model()

# 3. Definición de columnas exactas del modelo
COLUMNAS_MODELO = [
    'intensidad', 'HORA', 'TIPO_DIA_Festivo', 'TIPO_DIA_Laboral',
    'TIPO_DIA_Sabado', 'ESTACION_Barrio del Pilar',
    'ESTACION_Casa de Campo', 'ESTACION_Cuatro Caminos',
    'ESTACION_Ensanche de Vallecas', 'ESTACION_Escuelas Aguirre',
    'ESTACION_Farolillo', 'ESTACION_Juan Carlos I', 'ESTACION_Moratalaz',
    'ESTACION_Plaza Elíptica', 'ESTACION_Plaza de España',
    'ESTACION_Plaza del Carmen', 'T', 'VV', 'PB', 'RS', 'P'
]

# 4. Inicializar historial en el estado de la sesión
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Paso', 'Real_Simulado', 'Prediccion_Modelo'])
    st.session_state.step_counter = 0

# --- INTERFAZ LATERAL (INPUTS) ---
st.sidebar.header("Parámetros de Entrada")
estacion_sel = st.sidebar.selectbox("Estación", [
    "Moratalaz", "Barrio del Pilar", "Casa de Campo", "Cuatro Caminos", 
    "Ensanche de Vallecas", "Escuelas Aguirre", "Farolillo", 
    "Juan Carlos I", "Plaza Elíptica", "Plaza de España", "Plaza del Carmen"
])
tipo_dia_sel = st.sidebar.selectbox("Tipo de Día", ["Laboral", "Festivo", "Sabado"])
hora_sel = st.sidebar.slider("Hora del día", 0, 23, 12)

col_env1, col_env2 = st.sidebar.columns(2)
temp = col_env1.number_input("Temperatura (T)", value=23.0)
viento = col_env2.number_input("Viento (VV)", value=5.0)
presion = col_env1.number_input("Presión (PB)", value=1000.0)
radiacion = col_env2.number_input("Radiación (RS)", value=200.0)
precip = st.sidebar.number_input("Precipitación (P)", value=0.0)
intensidad = st.sidebar.number_input("Intensidad Tráfico", value=100)

# --- LÓGICA DE PREDICCIÓN ---
def realizar_prediccion():
    # Crear diccionario base con ceros
    input_dict = {col: 0 for col in COLUMNAS_MODELO}
    
    # Asignar valores numéricos
    input_dict['intensidad'] = intensidad
    input_dict['HORA'] = hora_sel
    input_dict['T'] = temp
    input_dict['VV'] = viento
    input_dict['PB'] = presion
    input_dict['RS'] = radiacion
    input_dict['P'] = precip
    
    # Activar One-Hot Encoding para Estación y Día
    col_est = f"ESTACION_{estacion_sel}"
    col_dia = f"TIPO_DIA_{tipo_dia_sel}"
    
    if col_est in input_dict: input_dict[col_est] = 1
    if col_dia in input_dict: input_dict[col_dia] = 1
    
    df_input = pd.DataFrame([input_dict])
    return model.predict(df_input)[0]

# --- DASHBOARD PRINCIPAL ---
st.title("Sistema de Predicción vs Realidad")

# Contenedor para que el gráfico no salte de posición
placeholder = st.empty()

# Generamos el nuevo punto
pred_actual = realizar_prediccion()
# Simulamos el valor real del API (un rango cercano a la prediccion para que tenga sentido)
valor_real_api = pred_actual + random.uniform(-10, 10) 

# Actualizar historial
nuevo_punto = pd.DataFrame({
    'Paso': [st.session_state.step_counter],
    'Real_Simulado': [valor_real_api],
    'Prediccion_Modelo': [pred_actual]
})

st.session_state.history = pd.concat([st.session_state.history, nuevo_punto], ignore_index=True).tail(30)
st.session_state.step_counter += 1

with placeholder.container():
    # Gráfico de líneas
    st.line_chart(st.session_state.history.set_index('Paso'))
    
    # Métricas visuales
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicción Actual", f"{pred_actual:.2f}")
    m2.metric("Valor Real (API)", f"{valor_real_api:.2f}")
    m3.metric("Diferencia (Error)", f"{abs(pred_actual - valor_real_api):.2f}")

# Control de tiempo
st.info("La gráfica se actualiza cada 5 segundos reflejando los cambios en los parámetros.")
time.sleep(5)
st.rerun()


