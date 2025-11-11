import streamlit as st
import pandas as pd
import joblib

model = joblib.load("titanic_model.pkl")

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
st.title(" Titanic Survival Prediction")
st.write("Enter passenger details below to predict survival chances.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid (in $)", 0.0, 600.0, 30.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

sex_val = 0 if sex == "Female" else 1
embarked_val = {"C": 0, "Q": 1, "S": 2}[embarked]

data = pd.DataFrame([{
    "PassengerId": 999,
    "Pclass": pclass,
    "Sex": sex_val,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked_val
}])

if st.button("Predict Survival"):
    result = model.predict(data)[0]
    if result == 1:
        st.success(" The passenger would have survived!")
    else:
        st.error("The passenger would not have survived.")
