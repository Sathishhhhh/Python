import pandas as pd

## 2. successful_bookings value is greater than capacity 


df4 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/fact_aggregated_bookings.csv")

df4[df4["successful_bookings"] > df4["capacity"]]




(df4["successful_bookings"] > df4["capacity"]).sum()



invalid_bookings = df4[df4["successful_bookings"] > df4["capacity"]]
print(invalid_bookings.head())





print(df4)

