# app/app.py
import streamlit as st
from predict import predict

st.title("Iris Flower Classifier")
st.write("Enter the measurements to predict the Iris species.")

# User inputs
petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=2.0)
petal_width = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=2.0)
sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=4.0)
sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=4.0)

if st.button("🔮 Predict"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    species = predict(features)
    st.success(f"Predicted Iris species: **{species}**")
