import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

fraud = df["isFraud"].value_counts()

plt.figure(figsize=(6,4))

plt.bar(["Legitimate", "Fraud"], fraud)

plt.title("Fraud Distribution")

plt.xlabel("Transaction Type")

plt.ylabel("Count")

plt.show()