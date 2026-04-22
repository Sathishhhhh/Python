# 🏥 Healthcare Premium Prediction

A machine learning regression project that predicts healthcare insurance premiums based on patient characteristics.

## 📊 Project Overview

This project uses regression algorithms to predict annual healthcare insurance premiums based on factors such as:
- Age
- Sex
- BMI (Body Mass Index)
- Number of Children
- Smoking Status
- Region

## 🎯 Models Implemented

1. **Linear Regression** - Baseline model for understanding linear relationships
2. **Random Forest Regressor** - Ensemble method capturing non-linear patterns
3. **Gradient Boosting Regressor** - Advanced ensemble technique for improved accuracy

## 📁 Project Structure

```
Project 4 - Healthcare Premium Prediction/
├── data/
│   └── generate_sample_data.py       # Generate sample dataset
├── models/                            # Saved model files
├── preprocessing.py                   # Data preprocessing pipeline
├── train_model.py                     # Model training script
├── app.py                            # Streamlit web interface
├── requirements.txt                  # Project dependencies
└── README.md                         # This file
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data
```bash
python data/generate_sample_data.py
```

### 3. Train Models
```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train all three regression models
- Evaluate performance metrics
- Save the best performing model

### 4. Run Streamlit App
```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

## 📈 Model Performance Metrics

The models are evaluated using:
- **RMSE (Root Mean Squared Error)** - Lower is better
- **MAE (Mean Absolute Error)** - Average prediction error in dollars
- **R² Score** - Coefficient of determination (0-1, higher is better)

## 💻 Features

### Interactive Web Interface
- Slider inputs for continuous variables (age, BMI)
- Dropdown selections for categorical variables (sex, region, smoker)
- Real-time premium prediction
- Patient summary display
- Health insights based on input parameters

### Machine Learning Pipeline
- Automatic data loading and validation
- Categorical variable encoding
- Feature scaling and normalization
- Train-test split for model evaluation
- Model serialization for deployment

## 📊 Key Insights

- **Smokers** typically pay significantly higher premiums
- **Age** is a strong predictor of premium costs
- **BMI** shows correlation with increased premiums
- **Region** may have regional variations in premiums
- **Family size** (children) impacts total healthcare costs

## 🔧 Customization

### Modify Model Parameters
Edit `train_model.py` to adjust:
- Number of trees in Random Forest
- Learning rate in Gradient Boosting
- Train-test split ratio

### Update Feature Ranges
Edit `app.py` to change input slider ranges and selections

### Add More Features
1. Update `data/generate_sample_data.py` to include new features
2. Modify `preprocessing.py` to handle new variables
3. Update `app.py` to accept new input parameters

## 📝 Data Format

The healthcare dataset should contain:
```
age,sex,bmi,children,smoker,region,charges
35,male,24.5,2,no,northeast,11000.50
42,female,27.3,1,yes,southeast,25000.75
```

## 🎓 Learning Objectives

This project demonstrates:
- Data preprocessing and feature engineering
- Multiple regression algorithms
- Model evaluation and comparison
- Hyperparameter tuning
- Web application development with Streamlit
- Model serialization and deployment

## 📚 Technologies Used

- **Python** - Programming language
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning algorithms
- **Streamlit** - Web interface framework
- **NumPy** - Numerical computations

## 🤝 Contributing

Feel free to:
- Add more features to the dataset
- Implement additional algorithms
- Improve the UI/UX
- Add data visualization
- Enhance model accuracy

## 📄 License

This project is open-source and available for educational purposes.

## 🎯 Next Steps

1. Deploy the Streamlit app using Streamlit Cloud or Heroku
2. Integrate with real healthcare data
3. Add data visualization dashboards
4. Implement cross-validation
5. Add model explainability (SHAP values)

---

**Created:** 2026 | **Version:** 1.0 | **Status:** Active
