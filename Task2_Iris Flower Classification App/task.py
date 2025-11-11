import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Iris Flower Classifier",
    layout="centered"
)

st.title(" Iris Flower Classification App")
st.write("""
Welcome to the **Iris Flower Classifier**!  
This app uses a trained **Support Vector Machine (SVM)** model to predict the 
species of an Iris flower based on its **sepal and petal measurements**.
""")

@st.cache_resource
def load_model():
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_model()
    st.sidebar.success(" Model Loaded Successfully")
except FileNotFoundError:
    st.error("""
Model files not found!  
Please make sure the following files are in the same folder:
- `svm_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
    """)
    st.stop()

st.subheader(" Enter Flower Measurements")
st.write("Provide the flower's details below:")

sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2)

if st.button(" Predict Species"):
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_species = label_encoder.inverse_transform(prediction)[0]

    st.success(f" Predicted Iris species: **{predicted_species}**")

st.sidebar.header("About This App")
st.sidebar.write("""
- **Algorithm:** Support Vector Machine (SVM)  
- **Dataset:** Iris (3 species: Setosa, Versicolor, Virginica)  
- **Created By:** Ritesh Wadikar  
""")
st.sidebar.caption("Built with  using Streamlit")
