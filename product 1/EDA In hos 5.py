import pandas as pd

df5 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/fact_bookings.csv")
df4 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/fact_aggregated_bookings.csv")


## Example merge
df_all = pd.merge(
    df5,
    df4,
    on="property_id")



# Example aggregation


revenue_by_platform = df_all.groupby("booking_platform")["revenue_realized"].sum()

print(revenue_by_platform)



print(df_all)