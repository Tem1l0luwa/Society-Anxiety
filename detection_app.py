#load the necessary libraries
import streamlit as st
import pandas as pd
import pickle

#load the model
with open('social_anxiety.pkl', 'rb') as f:
    model= pickle.load(f)
    
    
#streamlit app
st.title("Detecting Social Anxiety App")
st.write("Enter the details below to detect the social anxiety level of the client")

# collect the input from the client
age = st.number_input("Age", min_value=1, step=1, value=20)
age_cat = st.selectbox("Age Category", ["Teen", "Adult", "Senior"], index=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=1)
occupation = st.text_input("Occupation", value="Data Scientist")
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, step=0.5, value=5.0)
physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0.0, step=0.5, value=2.0)
caffeine_intake = st.number_input("Caffeine Intake (mg/day)", min_value=0, step=50, value=0)
alcohol_consumption = st.number_input("Alcohol Consumption (drinks/week)", min_value=0, step=1, value=0)
smoking = st.selectbox("Smoking", ["Yes", "No"], index=1)
family_history = st.selectbox("Family History of Anxiety", ["Yes", "No"], index=1)
stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=9)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, step=1, value=80)
breathing_rate = st.number_input("Breathing Rate (breaths/min)", min_value=5, max_value=50, step=1, value=20)
sweating_level = st.slider("Sweating Level (1-5)", min_value=1, max_value=5, value=3)
dizziness = st.selectbox("Dizziness", ["Yes", "No"], index=1)
medication = st.selectbox("Medication", ["Yes", "No"], index=1)
therapy_sessions = st.number_input("Therapy Sessions (per month)", min_value=0, step=1, value=0)
recent_life_event = st.selectbox("Recent Major Life Event", ["Yes", "No"], index=1)
diet_quality = st.slider("Diet Quality (1-10)", min_value=1, max_value=10, value=5)

# detection button
if st.button("Detect Social Anxiety Level"):
    data = {
        'Age': [age],
        'Age_cat': [age_cat],
        'Gender': [gender],
        'Occupation': [occupation],
        'Sleep Hours': [sleep_hours],
        'Physical Activity (hrs/week)': [physical_activity],
        'Caffeine Intake (mg/day)': [caffeine_intake],
        'Alcohol Consumption (drinks/week)': [alcohol_consumption],
        'Smoking': [smoking],
        'Family History of Anxiety': [family_history],
        'Stress Level (1-10)': [stress_level],
        'Heart Rate (bpm)': [heart_rate],
        'Breathing Rate (breaths/min)': [breathing_rate],
        'Sweating Level (1-5)': [sweating_level],
        'Dizziness': [dizziness],
        'Medication': [medication],
        'Therapy Sessions (per month)': [therapy_sessions],
        'Recent Major Life Event': [recent_life_event],
        'Diet Quality (1-10)': [diet_quality]
    }
    input_df= pd.DataFrame(data)
    
    #predict
    detection= model.predict(input_df)[0]
    
    #Display the result
    st.success(f"The Level of Anxiety is {detection}")