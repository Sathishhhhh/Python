import pandas as pd
import numpy as np

np.random.seed(42)

# Generate credit risk dataset
n_samples = 500

data = {
    'Age': np.random.randint(18, 80, n_samples),
    'Income': np.random.randint(15000, 200000, n_samples),
    'Credit_Score': np.random.randint(300, 850, n_samples),
    'Loan_Amount': np.random.randint(5000, 500000, n_samples),
    'Loan_Tenure_Years': np.random.randint(1, 30, n_samples),
    'Employment_Years': np.random.randint(0, 40, n_samples),
    'Num_Credit_Accounts': np.random.randint(1, 10, n_samples),
    'Num_Defaults_Past': np.random.randint(0, 5, n_samples),
    'Monthly_Debt_Payment': np.random.randint(0, 5000, n_samples),
    'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
    'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
}

df = pd.DataFrame(data)

# Create target variable with realistic patterns
# Higher risk factors: low credit score, high debt, low income, multiple defaults
default_risk = (
    (df['Credit_Score'] < 500).astype(int) * 0.4 +
    (df['Num_Defaults_Past'] > 0).astype(int) * 0.3 +
    ((df['Monthly_Debt_Payment'] / (df['Income'] / 12)) > 0.5).astype(int) * 0.3 +
    (df['Loan_Amount'] > df['Income'] * 3).astype(int) * 0.2 +
    np.random.uniform(0, 0.3, n_samples)
)

df['Default'] = (default_risk > 0.5).astype(int)

# Balance classes slightly (more defaults in real data, but not extreme)
df.to_csv('credit_risk_data.csv', index=False)
print(f"Credit Risk Dataset Generated: {df.shape[0]} records")
print(f"\nClass Distribution:")
print(df['Default'].value_counts())
print(f"\nDefault Rate: {df['Default'].mean():.2%}")
print(f"\nFirst few records:")
print(df.head(10))
