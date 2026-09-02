import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

transaction_types = df["type"].value_counts()

plt.figure(figsize=(8,5))

transaction_types.plot(kind="bar", color="skyblue")

plt.title("Transaction Types")

plt.xlabel("Type")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.show()