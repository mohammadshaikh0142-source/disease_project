import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import os
import numpy as np
working_dir = os.getcwd()
model_heart = pickle.load(open(f'heart_model.pkl','rb'))
scaler_heart = pickle.load(open(f'heart_scaler.pkl','rb'))

model_diabetes = pickle.load(open(f'diabetes_model.pkl','rb'))
scaler_diabetes = pickle.load(open(f'diabetes_scaler.pkl','rb'))

model_hypertension = pickle.load(open(f'Hypertension_model.pkl','rb'))
scaler_hypertension = pickle.load(open(f'Hypertension_scaler.pkl','rb'))

model_lungcancer = pickle.load(open(f'lung_cancer_model.pkl','rb'))
scaler_lungcaner = pickle.load(open(f'lung_cancer_scaler.pkl','rb'))

with st.sidebar:
    select = option_menu('Disease Prediction Using Machine_lerning',
                        ['Description','Heart_disease','Diabetes (Sugar)','Hypertension (BP)','Lung Cancer'],
                        icons=['chart-left-text','suit-heart','activity','droplet-half','lungs'],default_index=0)


if select=='Description':
    st.header(':red[Disease_Prediction using Machine_Learning]')
    st.caption('''
Project Description:
This project aims to build an intelligent machine learning system that can predict the diseases based on patient health dataThe system takes health parameters such as age, gender, blood pressure, glucose level, cholesterol, BMI, and lifestyle factors as input and predicts the probability of diseases like diabetes, heart disease, hypertension, and lung cancer.
Project usecase:\n
Predict the risk of multiple diseases from a single platform.\n
Improve healthcare accessibility by providing early warning .\n
Assist doctors and patients with data-driven decision support.\n
An interactive user interface for input and output visualization.\n
A scalable system that can be expanded to include more diseases in the future.''')

if select=='Heart_disease':
    st.title(':red[Heart Disease Prediction]')

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Enter the Age',12,100)
        sex = st.selectbox('Gender',['Male','Female'])
        cp = st.selectbox('Chest Pain types',['Typical angina',
                'Atypical angina','Non-anginal pain','Asymptomatic'])
        trestbps = st.number_input('what is your resting Blood Pressure',90,200)
        chol = st.number_input('Serum Cholestoral level in mg/dl',100,600)
        fbs = st.selectbox('Is your body glucose level is less than 120 mg/dl',['Yes','No'])

    with col2:
        restecg = st.selectbox('Resting Electrocardiographic results',['Normal','Abnormal'])
        thalach = st.number_input('Maximum Heart Rate achieved',60,210)
        exang = st.selectbox('hest pain (angina) that happens when you exercise',['No angina during exercise',
                'Angina present during exercise'])
        oldpeak = st.number_input('ST depression induced by exercise (reduced blood flow to the heart)',0.0,7.0)
        slope = st.selectbox('Slope of the peak exercise ST segment (ST segment on an ECG at the highest point of exercise)',['Upsloping','Flat','Downsloping'])
        ca = st.number_input('Major vessels colored by (flourosopy 0 → No vessel colored. 1, 2, 3 → Number of vessels colored. 4 → Sometimes used to mean not applicable',0,5)
    thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect',0,4)


    sex = 1 if sex=='Male' else 0
    fbs = 0 if fbs=='Yes' else 1
    restecg = 0 if restecg=='Normal' else 1
    exang = 0 if exang=='No angina during exercise' else 1
    if cp=='Typical angina':
        cp = 0
    elif cp=='Atypical angina':
        cp = 1
    elif cp=='Non-anginal pain':
        cp = 2
    else:
        cp = 3
    if slope=='Upsloping':
        slope=0
    elif slope=='Flat':
        slope=1
    else:
        slope=2
    

    # code for Prediction
    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        try:
            inp1 = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca, thal]
            input_data1 = np.asarray(inp1)
            input_data_reshaped1 = input_data1.reshape(1,-1)
            input_data_std2 = scaler_heart.transform(input_data_reshaped1)
            heartd_prediction = model_heart.predict(input_data_std2)

            if heartd_prediction[0] == 1:
                heart_diagnosis = 'The patient can have a heart disease'
            else:
                heart_diagnosis = 'The patient does not have any heart disease'

        except:
            st.warning('fill all the value first')
    st.success(heart_diagnosis)
    
if select=='Diabetes (Sugar)':
    st.title(':red[Diabetes Prediction]')

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender',['Male','Female'])
        if  gender=='Female':
            Pregnancies = st.number_input('Enter Number of Pregnancies',0,12)
        else:
            Pregnancies=0
        Glucose = st.number_input('what is your glucose Level?',0,199)
        BloodPressure = st.number_input('what is your blood Pressure value?',0,150)
        SkinThickness = st.number_input('what is your skin Thickness value (millimeter)?',0,100)
    with col2:
        Insulin = st.number_input('what is your insulin Level?',0,900)
        BMI = st.number_input('BMI value',0.0,70.0)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value (it is the score that show likelihood of diabetes based on your family history)',0.001,3.00)
        Age = st.number_input('Enter your age',12,100)

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            inp2 = [Pregnancies,Glucose,BloodPressure,SkinThickness,
                Insulin,BMI,DiabetesPedigreeFunction,Age]
            input_data2 = np.asarray(inp2)
            input_data_reshaped1 = input_data2.reshape(1,-1)
            input_data_std1 = scaler_diabetes.transform(input_data_reshaped1)
            diab_prediction = model_diabetes.predict(input_data_std1)
            
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The patient is diabetic'
            else:
                diab_diagnosis = 'The patient is non diabetic'
        
        except:
            st.warning('fill all the value first')
    st.success(diab_diagnosis)


if select=='Hypertension (BP)':
    st.title(":red[Hypertension (BP) Prediction]")

    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox('Gender',['Male','Female'])
        age = st.number_input('Enter ypur Age',12,100)
        currentSmoker = st.selectbox('Do you smoke?',['Yes','No'])
        cigsPerDay = st.number_input('Number of cigrates you smoke per day',0,70)
        BPMeds = st.selectbox('Are you taking BP medicines?',['Yes','No'])
        diabetes = st.selectbox('Do you have diabetes?',['Yes','No'])

    with col2:
        totChol = st.number_input('what is your total cholestrol level?',100,700)
        sysBP = st.number_input('systolic blood pressure value (Example: In 120/80 mmHg)',80,300)
        diaBP = st.number_input('Distollic blood pressure (Example: In 120/80 mmHg)',40,160)
        BMI_bp = st.number_input('Body mass index value (BMI) [Formula: BMI = weight (kg) / [height (m)]²]',10,60)
        heartRate = st.number_input('what is your current heart rate?',40,160)
        glucose = st.number_input('what is your glucose level?',30,400)

    sex = 1 if sex=='Male' else 0
    currentSmoker = 1 if currentSmoker=='Yes' else 0
    BPMeds = 1 if BPMeds=='Yes' else 0
    diabetes = 1 if diabetes=='Yes' else 0
 
    # code for Prediction
    Hypertension_diagnosis = ''
    if st.button("Hypertension result"):
        try:
            inp3 = [sex,age,currentSmoker,cigsPerDay,BPMeds,diabetes,totChol,sysBP,
                    diaBP,BMI_bp,heartRate,glucose]
            input_data3 = np.asarray(inp3)
            input_data_reshaped3 = input_data3.reshape(1,-1)
            input_data_std3 = scaler_hypertension.transform(input_data_reshaped3)
            Hypertension_prediction = model_hypertension.predict(input_data_std3)

            if Hypertension_prediction[0] == 1:
                Hypertension_diagnosis = "The person has Hypertension(BP) problem"
            else:
                Hypertension_diagnosis = "The person does not have Hypertension(BP) problem"

        except:
            st.warning('fill all the value first')

    st.success(Hypertension_diagnosis)
if select=='Lung Cancer':
    st.title(":red[Lung caner Prediction]")

    col1, col2 = st.columns(2)
    with col1:
        GENDER = st.selectbox('Gender',['Male','Female'])
        age_lc = st.number_input('Enter your Age',12,100)
        SMOKING = st.selectbox('Do you smoke',['Yes','No'])
        YELLOW_FINGERS = st.selectbox('Do you have yellow fingures? (yellowish discoloration of the fingertips)',['Yes','No'])
        ANXIETY = st.selectbox('Are you facing anxiety problem',['Yes','No'])
        PEER_PRESSURE = st.selectbox('Do you have any type of peer pressure',['Yes','No'])
        CHRONIC_DISEASE = st.selectbox('Do you have any type of chronic disease',['Yes','No'])
        FATIGUE = st.selectbox('Do you feel tired/overworked',['Yes','No'])
       
    with col2:
        ALLERGY = st.selectbox('Do you have any type of allergy',['Yes','No'])
        WHEEZING = st.selectbox('Do you wheeze? (Wheezing is a high-pitched whistling sound made while breathing)',['Yes','No'])
        ALCOHOL_CONSUMING = st.selectbox('Do you consume alcohol?',['Yes','No'])
        COUGHING = st.selectbox('Are you facing problem of coughing',['Yes','No'])
        SHORTNESS_OF_BREATH = st.selectbox('Do you face problem of breath_shortness',['Yes','No'])
        SWALLOWING_DIFFICULTY = st.selectbox('Do you face difficulty in swallowing',['Yes','No'])
        CHEST_PAIN = st.selectbox('Do you feel any type of chest pain',['Yes','No'])

    GENDER = 1 if GENDER=='Male' else 2
    SMOKING = 2 if SMOKING=='Yes' else 1
    YELLOW_FINGERS = 2 if YELLOW_FINGERS=='Yes' else 1
    ANXIETY = 2 if ANXIETY=='Yes' else 1
    PEER_PRESSURE = 2 if PEER_PRESSURE=='Yes' else 1
    CHRONIC_DISEASE = 2 if CHRONIC_DISEASE=='Yes' else 1
    FATIGUE = 1 if FATIGUE=='Yes' else 2
    ALLERGY = 1 if ALLERGY=='Yes' else 2
    WHEEZING = 2 if WHEEZING=='Yes' else 1
    ALCOHOL_CONSUMING = 2 if ALCOHOL_CONSUMING=='Yes' else 1
    COUGHING = 2 if COUGHING=='Yes' else 1
    SHORTNESS_OF_BREATH = 2 if SHORTNESS_OF_BREATH=='Yes' else 1
    SWALLOWING_DIFFICULTY = 2 if SWALLOWING_DIFFICULTY=='Yes' else 1
    CHEST_PAIN = 2 if CHEST_PAIN=='Yes' else 1

    # code for Prediction
    Lung_cancer_diagnosis = ''
    if st.button("Lung cancer result"):
        try:
            inp4 = [GENDER, age_lc, SMOKING, YELLOW_FINGERS, ANXIETY,
        PEER_PRESSURE, CHRONIC_DISEASE,FATIGUE , ALLERGY , WHEEZING,ALCOHOL_CONSUMING, 
        COUGHING, SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY, CHEST_PAIN]
            input_data4 = np.asarray(inp4)
            input_data_reshaped4 = input_data4.reshape(1,-1)
            input_data_std4 = scaler_lungcaner.transform(input_data_reshaped4)
            Lung_cancer_prediction = model_lungcancer.predict(input_data_std4)

            if Lung_cancer_prediction[0] == 1:
                Lung_cancer_diagnosis = "The person can have lung caancer"
            else:
                Lung_cancer_diagnosis = "The person does not have lung cancer"

        except:
            st.warning('fill all the value first')

    st.success(Lung_cancer_diagnosis)



import google.generativeai as genai
api_key = 'AIzaSyA01aSleedXLG5JOykBV0jnBdl2SPmhW3w'
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


st.divider()
user_chat = st.text_input(':green[✧ chat with Ai to get additional detail for problem]')
if st.button('Enter'):
    try:
        if user_chat.lower() in ["exit", "quit", "thanks"]:
            print("Bot: Goodbye! 👋")

        else:
            response = model.generate_content(user_chat)
            st.write('Ai assistant:', response.text, '\n')
    except Exception as e:
        st.caption(f'An error occured(api key expired) {e}')

