import pandas as pd

df4 = pd.read_csv("/Users/sathishkumar/Downloads/source-code/3_project_hospitality_analysis/datasets/fact_aggregated_bookings.csv")

df4["occ_pct"] = (df4["successful_bookings"] / df4["capacity"]) * 100


print(df4["occ_pct"].head(5))

df4.to_csv("dadadadadad.csv")
