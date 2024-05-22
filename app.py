import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load preprocessing objects
scaler_Age_at_enrollment = joblib.load("model/scaler_Age_at_enrollment.joblib")
scaler_Previous_qualification_grade = joblib.load("model/scaler_Previous_qualification_grade.joblib")
scaler_Admission_grade = joblib.load("model/scaler_Admission_grade.joblib")
scaler_Curricular_units_1st_sem_evaluations = joblib.load("model/scaler_Curricular_units_1st_sem_evaluations.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_evaluations = joblib.load("model/scaler_Curricular_units_2nd_sem_evaluations.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("model/scaler_Curricular_units_2nd_sem_grade.joblib")
scaler_Curricular_units_1st_sem_enrolled = joblib.load("model/scaler_Curricular_units_1st_sem_enrolled.joblib")
scaler_Curricular_units_2nd_sem_enrolled = joblib.load("model/scaler_Curricular_units_2nd_sem_enrolled.joblib")
encoder_Application_mode = joblib.load("model/encoder_Application_mode.joblib")
encoder_Mothers_qualification = joblib.load("model/encoder_Mothers_qualification.joblib")
encoder_Course = joblib.load("model/encoder_Course.joblib")
encoder_Displaced = joblib.load("model/encoder_Displaced.joblib")
pca_1 = joblib.load("model/pca_1.joblib")
model = joblib.load("model/gboost_model.joblib")
result_target = joblib.load("model/encoder_target.joblib")

pca_numerical_columns = [
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved', 
    'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_evaluations', 
    'Previous_qualification_grade', 'Curricular_units_2nd_sem_approved', 
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_evaluations', 
    'Curricular_units_2nd_sem_enrolled', 'Admission_grade'
]

# Load the feature names used during model training
feature_names = ['Application_mode', 'Course', 'Mothers_qualification', 
                 'Displaced', 'Age_at_enrollment',   
                  'pc_1', 'pc_2', 'pc_3']

# Function for preprocessing the data
def data_preprocessing(data):
    data = data.copy()
    df = pd.DataFrame()
    
    # Scaling numerical columns
    df["Age_at_enrollment"] = scaler_Age_at_enrollment.transform(np.asarray(data["Age_at_enrollment"]).reshape(-1, 1))[0]

    # Encoding categorical columns
    df["Application_mode"] = encoder_Application_mode.transform(data["Application_mode"])
    df["Mothers_qualification"] = encoder_Mothers_qualification.transform(data["Mothers_qualification"])
    df["Course"] = encoder_Course.transform(data["Course"])
    df["Displaced"] = encoder_Displaced.transform(data["Displaced"])

    # PCA
    data["Previous_qualification_grade"] = scaler_Previous_qualification_grade.transform(np.asarray(data["Previous_qualification_grade"]).reshape(-1, 1))[0]
    data["Admission_grade"] = scaler_Admission_grade.transform(np.asarray(data["Admission_grade"]).reshape(-1, 1))[0]
    data["Curricular_units_1st_sem_evaluations"] = scaler_Curricular_units_1st_sem_evaluations.transform(np.asarray(data["Curricular_units_1st_sem_evaluations"]).reshape(-1, 1))[0]
    data["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data["Curricular_units_1st_sem_approved"]).reshape(-1, 1))[0]
    data["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data["Curricular_units_1st_sem_grade"]).reshape(-1, 1))[0]
    data["Curricular_units_2nd_sem_evaluations"] = scaler_Curricular_units_2nd_sem_evaluations.transform(np.asarray(data["Curricular_units_2nd_sem_evaluations"]).reshape(-1, 1))[0]
    data["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(data["Curricular_units_2nd_sem_approved"]).reshape(-1, 1))[0]
    data["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(data["Curricular_units_2nd_sem_grade"]).reshape(-1, 1))[0]
    data["Curricular_units_1st_sem_enrolled"] = scaler_Curricular_units_1st_sem_enrolled.transform(np.asarray(data["Curricular_units_1st_sem_enrolled"]).reshape(-1, 1))[0]
    data["Curricular_units_2nd_sem_enrolled"] = scaler_Curricular_units_2nd_sem_enrolled.transform(np.asarray(data["Curricular_units_2nd_sem_enrolled"]).reshape(-1, 1))[0]

    df[["pc_1", "pc_2", "pc_3"]] = pca_1.transform(data[pca_numerical_columns])

    df = df[feature_names]

    return df

def prediction(data):
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result

# Streamlit UI
st.title('Early Dropout Detection for Jaya Jaya Institute')

# Input fields
data = pd.DataFrame()

# Create Streamlit form for user input
with st.form(key='dropout_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        Age_at_enrollment = st.number_input('Age at Enrollment', min_value=0, value=30)
        data["Age_at_enrollment"] = [Age_at_enrollment]

        Previous_qualification_grade = st.number_input('Previous Qualification Grade', min_value=0, value=0)
        data["Previous_qualification_grade"] = [Previous_qualification_grade]

        Admission_grade = st.number_input('Admission Grade', min_value=0, max_value=200, value=0)
        data["Admission_grade"] = [Admission_grade]

        Curricular_units_1st_sem_evaluations = st.number_input('Curricular Units 1st Sem Evaluations', min_value=0, value=0)
        data["Curricular_units_1st_sem_evaluations"] = [Curricular_units_1st_sem_evaluations]

        Curricular_units_1st_sem_approved = st.number_input('Curricular Units 1st Sem Approved', min_value=0, value=0)
        data["Curricular_units_1st_sem_approved"] = [Curricular_units_1st_sem_approved]

        Curricular_units_1st_sem_grade = st.number_input('Curricular Units 1st Sem Grade', min_value=0, value=0)
        data["Curricular_units_1st_sem_grade"] = [Curricular_units_1st_sem_grade]

    with col2:
        Curricular_units_2nd_sem_evaluations = st.number_input('Curricular Units 2nd Sem Evaluations', min_value=0, value=0)
        data["Curricular_units_2nd_sem_evaluations"] = [Curricular_units_2nd_sem_evaluations]

        Curricular_units_2nd_sem_approved = st.number_input('Curricular Units 2nd Sem Approved', min_value=0, value=0)
        data["Curricular_units_2nd_sem_approved"] = [Curricular_units_2nd_sem_approved]

        Curricular_units_2nd_sem_grade = st.number_input('Curricular Units 2nd Sem Grade', min_value=0, value=0)
        data["Curricular_units_2nd_sem_grade"] = [Curricular_units_2nd_sem_grade]

        Curricular_units_1st_sem_enrolled = st.number_input('Curricular Units 1st Sem Enrolled', min_value=0, value=0)
        data["Curricular_units_1st_sem_enrolled"] = [Curricular_units_1st_sem_enrolled]

        Curricular_units_2nd_sem_enrolled = st.number_input('Curricular Units 2nd Sem Enrolled', min_value=0, value=0)
        data["Curricular_units_2nd_sem_enrolled"] = [Curricular_units_2nd_sem_enrolled]

    Application_mode = st.selectbox('Application Mode', options=encoder_Application_mode.classes_, index=14)
    data["Application_mode"] = [Application_mode]

    Mothers_qualification = st.selectbox('Mother\'s Qualification', options=encoder_Mothers_qualification.classes_, index=9)
    data["Mothers_qualification"] = [Mothers_qualification]

    Course = st.selectbox('Course', options=encoder_Course.classes_, index=10)
    data["Course"] = [Course]

    Displaced = st.selectbox('Displaced', options=encoder_Displaced.classes_, index=0)
    data["Displaced"] = [Displaced]

    submit_button = st.form_submit_button(label='Predict Status')

with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)
    
if submit_button:
    new_data = data_preprocessing(data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Status Prediction: {}".format(prediction(new_data)))

st.caption('Copyright © 2023 Azel Rizki Nasution')