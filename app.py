import streamlit as st
import pandas as pd
import joblib
import time
import requests # Para las llamadas al API

# Configuración de la página
st.set_page_config(page_title="Predicciones vs Realidad", layout="wide")

# 1. Cargar el modelo (con cache para no ralentizar)
@st.cache_resource
def load_model():
    return joblib.load('modelo_rf.pkl')

model = load_model()

# 2. Inicializar el historial en la sesión de Streamlit
# Esto evita que los datos se borren cada vez que la app se refresca
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Paso', 'Real', 'Predicho'])
    st.session_state.step_counter = 0

st.title("Monitoreo en Tiempo Real: API vs Random Forest")

# Contenedor para el gráfico
placeholder = st.empty()

# 3. Bucle de actualización
while True:
    # Simulación de llamada al API (sustituye con tu URL real)
    # response = requests.get("https://api.tu-servicio.com/data")
    # data_api = response.json()['valor']
    
    # Datos de ejemplo para la demo
    import random
    valor_real_api = random.uniform(10, 20) 
    
    # 4. Predicción del modelo
    # Asegúrate de pasar los features que tu modelo espera
    features = pd.DataFrame([[valor_real_api]], columns=['Feature_Ejemplo'])
    prediccion = model.predict(features)[0]

    # 5. Actualizar el historial
    new_data = pd.DataFrame({
        'Paso': [st.session_state.step_counter],
        'Real': [valor_real_api],
        'Predicho': [prediccion]
    })
    
    st.session_state.history = pd.concat([st.session_state.history, new_data], ignore_index=True)
    st.session_state.step_counter += 1

    # Mantener solo los últimos 50 puntos para no saturar el gráfico
    st.session_state.history = st.session_state.history.tail(50)

    # 6. Graficar en el placeholder
    with placeholder.container():
        st.line_chart(st.session_state.history.set_index('Paso')[['Real', 'Predicho']])
        
        col1, col2 = st.columns(2)
        col1.metric("Valor Real (API)", f"{valor_real_api:.2f}")
        col2.metric("Predicción RF", f"{prediccion:.2f}")

    # Esperar N segundos antes de la siguiente llamada
    time.sleep(5) 
    
    # Forzar el refresco de la app
    st.rerun()