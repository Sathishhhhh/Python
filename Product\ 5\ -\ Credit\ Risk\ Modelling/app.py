import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💳 Credit Risk Classification Model")
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
    # Sidebar for navigation
    page = st.sidebar.radio("Navigation", ["🔮 Risk Assessment", "📊 Risk Analytics", "ℹ️ About"])
    
    scaler = preprocessor_data['scaler']
    encoders = preprocessor_data['encoders']
    
    if page == "🔮 Risk Assessment":
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Applicant Information")
            
            age = st.slider("Age", min_value=18, max_value=80, value=35, step=1)
            income = st.number_input("Annual Income ($)", min_value=15000, max_value=500000, value=50000, step=1000)
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700, step=10)
            
            st.divider()
            
            loan_amount = st.number_input("Loan Amount ($)", min_value=5000, max_value=500000, value=100000, step=5000)
            loan_tenure_years = st.slider("Loan Tenure (Years)", min_value=1, max_value=30, value=5, step=1)
            employment_years = st.slider("Employment Years", min_value=0, max_value=40, value=5, step=1)
            
            st.divider()
            
            num_credit_accounts = st.slider("Number of Credit Accounts", min_value=1, max_value=10, value=2, step=1)
            num_defaults_past = st.slider("Previous Defaults", min_value=0, max_value=5, value=0, step=1)
            monthly_debt = st.number_input("Monthly Debt Payment ($)", min_value=0, max_value=10000, value=500, step=100)
            
            st.divider()
            
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
        
        # Prepare input data
        input_data = {
            'Age': age,
            'Income': income,
            'Credit_Score': credit_score,
            'Loan_Amount': loan_amount,
            'Loan_Tenure_Years': loan_tenure_years,
            'Employment_Years': employment_years,
            'Num_Credit_Accounts': num_credit_accounts,
            'Num_Defaults_Past': num_defaults_past,
            'Monthly_Debt_Payment': monthly_debt,
            'Marital_Status': marital_status,
            'Education': education
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for col in ['Marital_Status', 'Education']:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)
        
        risk_probability = prediction_proba[0][1]  # Probability of default
        
        with col2:
            st.subheader("⚠️ Risk Assessment Result")
            
            # Risk gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_probability * 100,
                title={'text': "Default Risk %"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=70, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk classification
            st.markdown("---")
            if prediction == 0:
                st.success(f"✅ **APPROVED** - Low Risk (Probability: {risk_probability:.2%})")
                recommendation = "Application can be approved"
                color = "green"
            else:
                st.error(f"❌ **DECLINED** - High Risk (Probability: {risk_probability:.2%})")
                recommendation = "Application should be declined or require additional verification"
                color = "red"
            
            st.write(f"**Recommendation:** {recommendation}")
        
        # Financial Analysis
        st.markdown("---")
        st.subheader("💰 Financial Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            debt_to_income = (monthly_debt * 12) / income if income > 0 else 0
            st.metric("Debt-to-Income Ratio", f"{debt_to_income:.2%}")
        
        with col2:
            loan_to_income = loan_amount / income if income > 0 else 0
            st.metric("Loan-to-Income Ratio", f"{loan_to_income:.2f}x")
        
        with col3:
            monthly_income = income / 12
            remaining_after_debt = (monthly_income - monthly_debt) / monthly_income if monthly_income > 0 else 0
            st.metric("Income after Debt", f"{remaining_after_debt:.2%}")
        
        with col4:
            st.metric("Credit Score Status", "Good" if credit_score >= 700 else "Fair" if credit_score >= 650 else "Poor")
        
        # Risk factors
        st.markdown("---")
        st.subheader("🔍 Risk Factors Analysis")
        
        risk_factors = []
        
        if credit_score < 600:
            risk_factors.append("🔴 Very Low Credit Score (<600)")
        elif credit_score < 650:
            risk_factors.append("🟠 Low Credit Score (<650)")
        elif credit_score < 700:
            risk_factors.append("🟡 Fair Credit Score (<700)")
        
        if num_defaults_past > 0:
            risk_factors.append(f"🔴 Previous Defaults: {num_defaults_past}")
        
        if debt_to_income > 0.5:
            risk_factors.append("🔴 High Debt-to-Income Ratio (>50%)")
        elif debt_to_income > 0.35:
            risk_factors.append("🟡 Moderate Debt-to-Income Ratio (>35%)")
        
        if loan_to_income > 3:
            risk_factors.append("🔴 Loan exceeds 3x Annual Income")
        elif loan_to_income > 2:
            risk_factors.append("🟡 Loan exceeds 2x Annual Income")
        
        if employment_years < 2:
            risk_factors.append("🟡 Limited Employment History (<2 years)")
        
        if num_credit_accounts < 2:
            risk_factors.append("🟡 Limited Credit History (<2 accounts)")
        
        if not risk_factors:
            st.success("✅ No significant risk factors identified")
        else:
            for factor in risk_factors:
                st.write(factor)
    
    elif page == "📊 Risk Analytics":
        st.subheader("📈 Risk Distribution & Insights")
        
        # Load the original data for analytics
        try:
            df = pd.read_csv('data/credit_risk_data.csv')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Default distribution
                default_counts = df['Default'].value_counts().sort_index()
                fig1 = px.pie(
                    values=default_counts.values,
                    names=['Non-Default', 'Default'],
                    title='Default Distribution in Dataset',
                    color_discrete_map={'Non-Default': 'lightgreen', 'Default': 'lightcoral'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Age distribution
                fig2 = px.histogram(
                    df,
                    x='Age',
                    nbins=20,
                    title='Age Distribution',
                    color_discrete_sequence=['skyblue']
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Credit Score vs Default
                fig3 = px.scatter(
                    df,
                    x='Credit_Score',
                    y='Income',
                    color='Default',
                    title='Credit Score vs Income (colored by Default)',
                    color_discrete_map={0: 'green', 1: 'red'},
                    hover_data=['Num_Defaults_Past']
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # Loan Amount Distribution
                fig4 = px.box(
                    df,
                    y='Loan_Amount',
                    x='Default',
                    title='Loan Amount by Default Status',
                    color_discrete_sequence=['skyblue']
                )
                st.plotly_chart(fig4, use_container_width=True)
            
        except FileNotFoundError:
            st.warning("Dataset not found. Generate sample data first.")
    
    elif page == "ℹ️ About":
        st.subheader("About This Model")
        
        st.markdown("""
        ### 🎯 Purpose
        This Credit Risk Classification model predicts the likelihood of a customer defaulting on a loan 
        based on their financial and demographic characteristics.
        
        ### 📊 Model Information
        - **Task**: Binary Classification (Default / No Default)
        - **Features**: 11 financial and demographic variables
        - **Training Data**: 500 historical credit records
        - **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
        
        ### 🔍 Key Features
        1. **Financial Metrics**
           - Annual Income
           - Loan Amount
           - Monthly Debt Payment
           - Debt-to-Income Ratio
        
        2. **Credit History**
           - Credit Score
           - Number of Credit Accounts
           - Previous Defaults
        
        3. **Employment & Demographics**
           - Employment Years
           - Age
           - Education Level
           - Marital Status
        
        4. **Loan Details**
           - Loan Tenure (Years)
           - Loan-to-Income Ratio
        
        ### ⚠️ Risk Assessment Levels
        - **Green (0-30%)**: Low Risk - Approve
        - **Yellow (30-60%)**: Medium Risk - Review
        - **Red (60-100%)**: High Risk - Decline
        
        ### 📈 Model Performance
        The model is trained using ensemble methods (Random Forest, Gradient Boosting) 
        with balanced class weights to handle class imbalance and provide robust predictions.
        
        ### ⚖️ Disclaimer
        This model is for educational and demonstration purposes. 
        Actual credit decisions should incorporate additional factors and expert judgment.
        """)

else:
    st.error("Unable to load the model. Please ensure the model files exist.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>Credit Risk Classification System | Machine Learning Classification Model</p>"
    "<p style='font-size: 0.8em; color: gray;'>Powered by Scikit-learn & Streamlit</p>"
    "</div>",
    unsafe_allow_html=True
)
