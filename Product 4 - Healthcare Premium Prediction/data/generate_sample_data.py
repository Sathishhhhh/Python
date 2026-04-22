import pandas as pd
import numpy as np

np.random.seed(42)

# Generate sample healthcare premium data
n_samples = 300

data = {
    'age': np.random.randint(18, 80, n_samples),
    'sex': np.random.choice(['male', 'female'], n_samples),
    'bmi': np.random.uniform(18, 40, n_samples),
    'children': np.random.randint(0, 5, n_samples),
    'smoker': np.random.choice(['yes', 'no'], n_samples),
    'region': np.random.choice(['northeast', 'southeast', 'southwest', 'northwest'], n_samples),
    'charges': np.random.uniform(1000, 50000, n_samples)
}

df = pd.DataFrame(data)

# Add some correlation to make it more realistic
df['charges'] = (df['age'] * 200 + 
                 df['bmi'] * 300 + 
                 (df['smoker'] == 'yes').astype(int) * 15000 +
                 df['children'] * 500 +
                 np.random.normal(0, 2000, n_samples))

df.to_csv('healthcare_data.csv', index=False)
print(f"Sample data generated: {df.shape[0]} records")
print(df.head())
