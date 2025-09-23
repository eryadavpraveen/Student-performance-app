import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_model():
    with open("student_lr_final_model.pkl","rb") as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le    


def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities']=le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed


def predict_data(data):
    model, scaler, le = load_model()
    processing_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processing_data)
    return prediction

def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to  get a prediction for your performance")

    hour_studied = st.number_input("Hours Studied", min_value=1, max_value=13, value=5)   
    prev_score = st.number_input("Previous Score", min_value=20, max_value=100, value=50)
    extra_acti = st.selectbox("Extra curricular Activities",['Yes','No'])   
    sleep_hrs = st.number_input("Sleeping Hours", min_value=3, max_value=10, value=7)   
    no_paper = st.number_input("Number of Question Paper Solved", min_value=0,)   

    if st.button("Predict Your Score"):
        user_data={
            "Hours Studied":hour_studied,
            "Previous Scores":prev_score,
            "Extracurricular Activities":extra_acti,
            "Sleep Hours":sleep_hrs,
            "Sample Question Papers Practiced":no_paper
        }
        
        prediction = predict_data(user_data)
        st.success(f"Your Prediction result is: {prediction}")
    

if __name__ =="__main__":
            main()
