import pandas as pd


#EXE 1

# Load tables
df_hotels = pd.read_csv(
    "/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/dim_hotels.csv"
)

df_bookings = pd.read_csv(
    "/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/fact_bookings.csv"
)

# Merge
df_all = pd.merge(
    df_bookings,
    df_hotels,
    on="property_id",
    how="inner"
)

# Revenue by city
data = (
    df_all.groupby("city")["revenue_realized"]
          .sum()
          .round(2)
          .sort_values(ascending=False)
)

print(data)


##2 

data1 = ( df_all.groupby("city")["ratings_given"]).mean().round()

print(data1)


#3

data2 = (df_all.groupby("booking_platform")["revenue_realized"]).sum().plot(kind = "pie")

print(data2)