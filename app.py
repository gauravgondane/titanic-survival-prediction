import streamlit as st
import pandas as pd
import pickle

# Load the logistic regression model and scaler
with open("logistic_regression_model.pkl", "rb") as model_file:
    log_reg = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Title of the app
st.title("Titanic Survival Prediction")

# Input features
st.write("Enter the features to get the prediction:")

# Create input fields for the user to enter feature values
Pclass = st.selectbox('Pclass (1 = 1st; 2 = 2nd; 3 = 3rd)', [1, 2, 3])
Sex = st.selectbox('Sex (0 = Female; 1 = Male)', [0, 1])
Age = st.number_input('Age', min_value=0, max_value=100, value=30)
SibSp = st.number_input('SibSp (Number of siblings/spouses aboard)', min_value=0, max_value=10, value=0)
Parch = st.number_input('Parch (Number of parents/children aboard)', min_value=0, max_value=10, value=0)
Fare = st.number_input('Fare', min_value=0.0, max_value=1000.0, value=50.0)
Embarked = st.selectbox('Embarked (0 = C; 1 = Q; 2 = S)', [0, 1, 2])

# Create a prediction button
if st.button("Predict"):
    # Create a dataframe for the input values
    input_data = pd.DataFrame({
        'Pclass': [Pclass],
        'Sex': [Sex],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'Embarked': [Embarked]
    })

    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)

    # Get the prediction
    prediction = log_reg.predict(input_data_scaled)

    # Display the prediction result
    if prediction[0] == 1:
        st.success("The passenger is likely to survive.")
    else:
        st.error("The passenger is not likely to survive.")
