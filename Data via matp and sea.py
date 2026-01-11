import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_excel("/Users/sathishkumar/Downloads/Matplotlib_seaborn_resources/New Folder With Items/linechart.xlsx")


plt.figure(figsize=(12, 4))

plt.plot(data["Quarter"], data["Fridge"], color="blue", label="Fridge")
plt.plot(data["Quarter"], data["Dishwasher"], color="green", label="Dishwasher")
plt.plot(data["Quarter"], data["Washing Machine"], color="red", label="Washing machine")

plt.title("Product sales")
plt.xlabel("Quarter")
plt.ylabel("Million $")
plt.legend()
plt.show()



