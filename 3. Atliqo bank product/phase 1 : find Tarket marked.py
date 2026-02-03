import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df_cust = pd.read_csv("/Users/sathishkumar/Desktop/sathishviti1996/Python/3. Atliqo bank product/customers.csv")
df_cust = pd.read_csv("/Users/sathishkumar/Desktop/sathishviti1996/Python/3. Atliqo bank product/customers.csv")
df_cust = pd.read_csv("/Users/sathishkumar/Desktop/sathishviti1996/Python/3. Atliqo bank product/customers.csv")



import mysql.connector

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    passwd='root',
    database='e_master_card'
)

df_cust = pd.read_sql("SELECT * FROM customers", conn)
df_cust.head(3)



print("Customers data",df_cust.shape)
print("Credit Score data",df_cs.shape)
print("Transactions data",df_trans.shape)