import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Healthcare Premium Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare Premium Prediction")
st.markdown("---")

# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor_data = pickle.load(f)
        return model, preprocessor_data
    except FileNotFoundError:
        st.error("Model files not found! Please run train_model.py first.")
        return None, None

model, preprocessor_data = load_model_and_preprocessor()

if model is not None:
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Enter Patient Information")
        
        age = st.slider("Age", min_value=18, max_value=80, value=35)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.slider("BMI (Body Mass Index)", min_value=18.0, max_value=40.0, value=25.0, step=0.1)
        children = st.slider("Number of Children", min_value=0, max_value=4, value=0)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["northeast", "southeast", "southwest", "northwest"])
    
    # Prepare input data
    input_data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    scaler = preprocessor_data['scaler']
    encoders = preprocessor_data['encoders']
    
    for col in ['sex', 'smoker', 'region']:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    with col2:
        st.subheader("💰 Prediction Result")
        
        # Display prediction
        st.metric(
            label="Estimated Annual Premium",
            value=f"${prediction:,.2f}",
            delta=None
        )
        
        # Display input summary
        st.subheader("📊 Patient Summary")
        summary_data = {
            "Age": f"{age} years",
            "Sex": sex.capitalize(),
            "BMI": f"{bmi:.1f}",
            "Children": children,
            "Smoker": smoker.upper(),
            "Region": region.capitalize()
        }
        
        for key, value in summary_data.items():
            st.write(f"**{key}:** {value}")
    
    # Display analysis
    st.markdown("---")
    st.subheader("📈 Analysis & Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if smoker == "yes":
            st.warning("⚠️ Smoking Status: This significantly increases premium")
        else:
            st.success("✓ Non-smoker: Lower risk category")
    
    with col2:
        if bmi > 30:
            st.warning(f"⚠️ High BMI ({bmi:.1f}): Obesity category")
        elif bmi >= 25:
            st.info(f"ℹ️ BMI ({bmi:.1f}): Overweight category")
        else:
            st.success(f"✓ BMI ({bmi:.1f}): Normal weight")
    
    with col3:
        if age > 60:
            st.warning(f"⚠️ Age ({age}): Higher premiums with age")
        elif age < 30:
            st.success(f"✓ Age ({age}): Younger demographic - Lower risk")
        else:
            st.info(f"ℹ️ Age ({age}): Middle-aged demographic")

else:
    st.error("Unable to load the model. Please ensure the model files exist.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>Healthcare Premium Prediction System | Machine Learning Regression Model</p>"
    "<p style='font-size: 0.8em; color: gray;'>This is a predictive model based on historical data. "
    "Actual premiums may vary.</p>"
    "</div>",
    unsafe_allow_html=True
)
