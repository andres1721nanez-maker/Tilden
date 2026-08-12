import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Unified Project Metrics", layout="wide")
st.title("Unified Project Metrics Dashboard")

@st.cache_data
def get_tensor_data(k_layers=3):
    return {f"Layer {i+1}": np.random.rand(10, 10) for i in range(k_layers)}

tensors = get_tensor_data()

st.sidebar.header("CDC Parameters")
cdc_value = st.sidebar.slider("CDC Factor", 0.0, 10.0, 3.6)

st.header("Tensor Heatmaps (Reactivos a CDC)")
cols = st.columns(3)
for i, (name, data) in enumerate(tensors.items()):
    with cols[i]:
        st.subheader(name)
        fig = px.imshow(data * cdc_value, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

st.header("Calculated Metrics")
T = np.concatenate([d.flatten() for d in tensors.values()])
Em = np.sum(np.abs(T)) * cdc_value
Is = np.mean(T) / cdc_value
delta_P = np.max(T) - np.min(T)

m1, m2, m3 = st.columns(3)
m1.metric("Energía (Em)", f"{Em:.2f}")
m2.metric("Intensidad (Is)", f"{Is:.4f}")
m3.metric("Rango (ΔP)", f"{delta_P:.2f}")

st.header("Comparative Analysis (v1 vs v2)")
data = {
    "Metric": ["Frobenius", "Media", "Desv. Estándar", "Energía Total", "Máximo", "Mínimo"],
    "v1": [12.4, 0.52, 0.11, 450.2, 1.2, 0.01],
    "v2": [12.6, 0.54, 0.12, 455.8, 1.3, 0.02]
}
df = pd.DataFrame(data)
st.table(df)

st.info("Para ejecutar: streamlit run main.py --server.port 8000 --server.address 0.0.0.0")
