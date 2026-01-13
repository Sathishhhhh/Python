import  pandas as pd  

df = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/dim_date.csv")
df2 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/dim_hotels.csv")
df3 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/dim_rooms.csv")
df4 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/fact_aggregated_bookings.csv")
df5 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/fact_bookings.csv")
df6 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/new_data_august.csv")

#df4 = df4["property_id"].unique()

df5= df5[["property_id","revenue_generated", "revenue_realized"]]

df = df[["date", "mmm yy"]]

df4 = df4[["property_id", "capacity"]]


high_capacity = df4.sort_values(by="capacity", ascending=False)

print(high_capacity.head(10))


max_capacity_property = df4.loc[df4["capacity"].idxmax()]

print(max_capacity_property)

high_capacity_props = df4[df4["capacity"] > 100]

print(high_capacity_props)


high_capacity_props.shape[0]

print(df4["capacity"].describe())






print(df4)
print(df5)
print(df.head(4))
print(df.tail(6))