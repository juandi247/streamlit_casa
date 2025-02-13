import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar el modelo y el scaler
modelo = joblib.load('modelo_peritazgo.bin')
scaler = joblib.load('scalercasa.bin')

# Título de la app
st.title("Predicción de Precio de Casas")

# Mostrar imagen debajo del título
st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScEz2kb3sQGsAA6czYC5nQ_2VO6wM2HXHUDQkfiA1wvNKqR2obNphc66JuJ5Sbceli2eI&usqp=CAU", use_column_width=True)

# Sidebar para ingresar datos
st.sidebar.header("Ingresar Datos de la Casa")

distancia = st.sidebar.number_input("Distancia (km)", min_value=0.0, value=10.0)
habitaciones = st.sidebar.number_input("Número de Habitaciones", min_value=1, value=3)
banos = st.sidebar.number_input("Número de Baños", min_value=1, value=2)
carros = st.sidebar.number_input("Número de Carros", min_value=1, value=2)
area_construida = st.sidebar.number_input("Área Construida (m²)", min_value=1.0, value=100.0)
area_no_construida = st.sidebar.number_input("Área No Construida (m²)", min_value=1.0, value=50.0)

# Crear el dataframe con los datos ingresados
data = {
    'Distancia': [distancia],
    'habitaciones': [habitaciones],
    'Banos': [banos],
    'Carros': [carros],
    'AreaCons': [area_construida],
    'AreaNoConstruida': [area_no_construida]
}

df_input = pd.DataFrame(data)

# Escalar los datos de entrada con el scaler cargado
df_scaled = scaler.transform(df_input)

# Realizar la predicción con el modelo
precio_predicho = modelo.predict(df_scaled)

# Mostrar el resultado de la predicción
st.header(f"El precio estimado de la casa es: ${precio_predicho[0]:,.2f}")
