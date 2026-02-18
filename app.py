import streamlit as st
import joblib
import urllib.request
import os

@st.cache_resource
def load_model():
    model_path = 'modelo_rf.pkl'
    
    # Si el archivo no está en el servidor de Streamlit, lo bajamos del Release de GitHub
    if not os.path.exists(model_path):
        # SUSTITUYE ESTA URL: Haz clic derecho en el archivo en tu 'Release' y dale a 'Copiar dirección de enlace'
        url = "https://github.com/TU_USUARIO/TU_REPO/releases/download/v1.0/modelo_rf.pkl"
        
        with st.spinner("Descargando modelo... un momento, por favor."):
            urllib.request.urlretrieve(url, model_path)
            
    return joblib.load(model_path)

# Ahora ya puedes usarlo normalmente
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
