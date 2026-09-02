import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

df = df.head(100000)

encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])

X = df[[
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]]

y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"{name} Metrics:")
    print("  Accuracy :", accuracy_score(y_test, y_pred))
    print("  Precision:", precision_score(y_test, y_pred))
    print("  Recall   :", recall_score(y_test, y_pred))
    print("  F1 Score :", f1_score(y_test, y_pred))
    
    # Save the model file
    filename = {
        "Random Forest": "models/rf_model.pkl",
        "Gradient Boosting": "models/gb_model.pkl",
        "Logistic Regression": "models/lr_model.pkl"
    }[name]
    
    joblib.dump(model, filename)
    print(f"Saved as {filename}")

print("\nAll models trained successfully!")