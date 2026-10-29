import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

try:
    _icon = Image.open('logo_mr.png')
except Exception:
    _icon = "✍️"
st.set_page_config(
    page_title="Detector de Dígitos · Miguel Ruiz",
    page_icon=_icon,
    layout="wide"
)

# ---- Estilo personalizado: paleta azul petróleo + terracota (Miguel Ruiz) ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700&family=Inter:wght@300;400;600&display=swap');
    .stApp {
        background: linear-gradient(135deg, #1b3a4b 0%, #14303d 100%);
        color: #e8eef0;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stToolbar"] { right: 1rem; }
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
    .stMarkdown, [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {
        color: #e8eef0 !important;
    }
    h1, h2, h3, h4 { font-family: 'Montserrat', sans-serif !important; color: #e07a5f !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #f2cc8f !important; }
    section[data-testid="stSidebar"] { background: #12272f; }
    section[data-testid="stSidebar"] * { color: #e8eef0 !important; }
    .stButton>button {
        background-color: #e07a5f; color: #fff !important; border-radius: 10px; border: none; font-weight: 600;
    }
    .stButton>button:hover { background-color: #c9583c; color: #fff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Mostrar versión en sidebar
with st.sidebar:
    st.markdown("<h2 style=\"font-family:'Montserrat',sans-serif; color:#f2cc8f;\">Miguel Ruiz</h2>", unsafe_allow_html=True)
    st.write(f"**TensorFlow:** {tf.__version__}")
    st.write(f"**Keras:** {keras.__version__}")

st.title("✍️ Detección de Dígitos Escritos a Mano")
st.markdown("**Modelo CNN entrenado con MNIST** · por Miguel Ruiz")

# Cargar modelo
@st.cache_resource
def load_mnist_model():
    """Carga el modelo MNIST desde archivo .keras"""
    
    # Buscar archivo del modelo
    model_files = ['mnist_model.keras', 'mnist_model.h5']
    
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                st.info(f"📂 Intentando cargar: {model_path}")
                model = keras.models.load_model(model_path)
                st.success(f"✅ Modelo cargado exitosamente desde: `{model_path}`")
                return model
            except Exception as e:
                st.error(f"❌ Error al cargar {model_path}: {str(e)}")
                continue
    
    # Si no se encuentra el modelo
    st.error("❌ No se encontró ningún modelo")
    st.info("""
    **Instrucciones:**
    1. Entrena el modelo en Google Colab
    2. Descarga `mnist_model.keras`
    3. Súbelo a la raíz de tu repositorio
    4. Reinicia la aplicación
    """)
    st.stop()

# Cargar modelo
model = load_mnist_model()

# Interfaz
st.write("### 🎨 Dibuja un dígito del 0 al 9")

col1, col2 = st.columns([2, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.write("### 🎮 Controles")
    predict_btn = st.button("🔍 **Predecir**", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Limpiar", use_container_width=True)
    
    if clear_btn:
        st.rerun()

# Predicción
if predict_btn:
    if canvas_result.image_data is not None:
        if np.max(canvas_result.image_data) == 0:
            st.warning("⚠️ Por favor, dibuja un dígito primero")
        else:
            with st.spinner("🤔 Analizando..."):
                # Procesar imagen
                image = Image.fromarray(
                    canvas_result.image_data.astype("uint8")
                ).convert("L")
                
                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(image_resized) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                # Predicción
                prediction = model.predict(img_array, verbose=0)
                digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100
            
            # Resultados
            st.success(f"## 🎯 Dígito detectado: **{digit}**")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Predicción", digit)
            with col_m2:
                st.metric("Confianza", f"{confidence:.1f}%")
            with col_m3:
                alternative = np.argsort(prediction[0])[-2]
                st.metric("2ª opción", alternative)
            
            # Mostrar imágenes
            st.write("### 🖼️ Procesamiento")
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.write("**Tu dibujo original:**")
                st.image(canvas_result.image_data, width=200)
            
            with col_img2:
                st.write("**Imagen procesada (28x28):**")
                st.image(image_resized, width=200)
            
            # Gráfico de probabilidades
            st.write("### 📊 Probabilidades por dígito")
            import pandas as pd
            prob_df = pd.DataFrame({
                'Dígito': [str(i) for i in range(10)],
                'Probabilidad (%)': prediction[0] * 100
            })
            st.bar_chart(prob_df.set_index('Dígito'))
            
            with st.expander("🔍 Ver detalles"):
                for i, prob in enumerate(prediction[0]):
                    emoji = "🎯" if i == digit else ""
                    st.write(f"{emoji} **Dígito {i}**: {prob*100:.2f}%")
    else:
        st.warning("⚠️ Por favor, dibuja un dígito en el canvas")

