import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])

plt.figure(figsize=(10,8))

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()