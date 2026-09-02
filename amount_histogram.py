import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

plt.figure(figsize=(10,5))

plt.hist(df["amount"], bins=50, color="orange", edgecolor="black")

plt.title("Transaction Amount Distribution")

plt.xlabel("Amount")

plt.ylabel("Frequency")

plt.grid(True)

plt.show()