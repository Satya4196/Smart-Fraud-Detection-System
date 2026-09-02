import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

count = len(df)

plt.figure(figsize=(5,5))

plt.bar(["Transactions"], [count], color="green")

plt.title("Total Transactions")

plt.ylabel("Count")

plt.show()