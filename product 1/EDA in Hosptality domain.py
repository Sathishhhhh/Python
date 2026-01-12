import  pandas as pd  

df = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/dim_date.csv")
df2 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/dim_hotels.csv")
df3 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/dim_rooms.csv")
df4 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/fact_aggregated_bookings.csv")
df5 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/fact_bookings.csv")
df6 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/new_data_august.csv")

data = df5.room_category.unique()

data = df5["booking_platform"].unique()

data  = df5["booking_platform"].describe

data5 = df5.revenue_generated.mix(),df5.revenue_generated.max()

print(df5["booking_platform"].value_counts().plot(kind="bar"))

print(data)
print(data5)