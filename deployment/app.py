import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Purchase Prediction App")
st.write("Model loaded successfully!")
This application predicts the likelihood of a customer purchasing the Wellness Tourism Package.
Enter customer details below to get a prediction.
""")

# Input widgets for features
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited", ""], index=2) # Default empty
age = st.number_input("Age", 18, 100, 30)
city_tier = st.number_input("City Tier", 1, 3, 1)
duration_of_pitch = st.number_input("Duration Of Pitch (minutes)", 0, 60, 10)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Unemployed", ""], index=4) # Default empty
gender = st.selectbox("Gender", ["Male", "Female", ""], index=2) # Default empty
number_of_person_visiting = st.number_input("Number of Persons Visiting", 1, 6, 2)
number_of_followups = st.number_input("Number of Follow-ups", 0, 10, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King", ""], index=5) # Default empty
preferred_property_star = st.number_input("Preferred Property Star", 1, 5, 3)
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", ""], index=3) # Default empty
number_of_trips = st.number_input("Number of Trips", 0, 20, 5)
passport = st.selectbox("Has Passport", [0, 1], index=0) # 0 for No, 1 for Yes
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.selectbox("Owns Car", [0, 1], index=0) # 0 for No, 1 for Yes
number_of_children_visiting = st.number_input("Number of Children Visiting", 0, 5, 0)
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP", ""], index=5) # Default empty
monthly_income = st.number_input("Monthly Income", 10000, 1000000, 50000)

input_data = pd.DataFrame([{
    "TypeofContact": type_of_contact,
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

if st.button("Predict Purchase"): # Changed button text
    prediction = model.predict(input_data)[0]
    result = "Customer will purchase the package" if prediction == 1 else "Customer will NOT purchase the package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
