import pandas as pd


df4 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/fact_aggregated_bookings.csv")

## 1. null values


df4.isna().sum()

df4.isna().sum()[df4.isna().sum() > 0]

numeric_cols = df4.select_dtypes(include='number').columns

df4[numeric_cols] = df4[numeric_cols].fillna(
    df4[numeric_cols].mean()
)


df4.isna().sum()

df4.fillna(df4.mean(numeric_only=True), inplace=True)




print(df4)
