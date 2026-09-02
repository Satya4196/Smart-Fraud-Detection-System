import csv
import io
import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
import seaborn as sns
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "PS_20174392719_1491204439457_log.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "PS_20174392719_1491204439457_log.csv"

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
CHART_DIR = BASE_DIR / "static" / "images" / "charts"
SAMPLE_ROWS = 100000
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TYPE_MAPPING = {name: index for index, name in enumerate(TRANSACTION_TYPES)}
TYPE_DISPLAY_NAMES = {
    "CASH_IN": "Receive Cash",
    "CASH_OUT": "Withdraw Cash",
    "DEBIT": "Bank Debit",
    "PAYMENT": "Make Payment",
    "TRANSFER": "Send Money",
}
FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# Simple in-memory user store (no DB required)
USERS = {
    "admin": {"password": "admin123", "role": "admin", "name": "Admin User"},
    "demo": {"password": "demo123", "role": "user", "name": "Demo User"},
}

app = Flask(__name__)
app.secret_key = "fraud-detection-system-secret-2024"

MODEL_PATHS = {
    "Random Forest": BASE_DIR / "models" / "rf_model.pkl",
    "Gradient Boosting": BASE_DIR / "models" / "gb_model.pkl",
    "Logistic Regression": BASE_DIR / "models" / "lr_model.pkl",
}
models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
model = models["Random Forest"]


# ─── Auth helpers ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated


# ─── Data helpers ─────────────────────────────────────────────────────────────
def load_dataset():
    return pd.read_csv(DATA_PATH, nrows=SAMPLE_ROWS).copy()


def save_chart(figure, filename):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHART_DIR / filename
    figure.savefig(output_path, bbox_inches="tight", dpi=140)
    plt.close(figure)
    return f"images/charts/{filename}"


def build_dashboard_assets():
    df = load_dataset()
    numeric_df = df.copy()
    numeric_df["type"] = numeric_df["type"].map(TYPE_MAPPING)

    plt.style.use("seaborn-v0_8-whitegrid")

    chart_paths = {}

    fraud_counts = df["isFraud"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=["Legitimate", "Fraud"],
        y=fraud_counts.values,
        palette=["#23b5d3", "#f35b6b"],
        ax=ax,
    )
    ax.set_title("Fraud Distribution")
    ax.set_xlabel("Transaction Class")
    ax.set_ylabel("Number of Transactions")
    chart_paths["fraud_distribution"] = save_chart(fig, "fraud_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    transaction_order = df["type"].value_counts().index
    sns.countplot(data=df, x="type", order=transaction_order, palette="Blues_r", ax=ax)
    ax.set_title("Transaction Types")
    ax.set_xlabel("Type")
    ax.set_ylabel("Count")
    chart_paths["transaction_types"] = save_chart(fig, "transaction_types.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.histplot(df["amount"], bins=40, color="#6f42c1", kde=True, ax=ax)
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency")
    chart_paths["amount_histogram"] = save_chart(fig, "amount_histogram.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    correlation = numeric_df[FEATURE_COLUMNS + ["isFraud"]].corr(numeric_only=True)
    sns.heatmap(correlation, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    chart_paths["heatmap"] = save_chart(fig, "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    avg_amount = (
        df.groupby("type", as_index=False)["amount"]
        .mean()
        .sort_values("amount", ascending=False)
    )
    sns.barplot(data=avg_amount, x="type", y="amount", palette="magma", ax=ax)
    ax.set_title("Average Amount by Transaction Type")
    ax.set_xlabel("Type")
    ax.set_ylabel("Average Amount")
    chart_paths["avg_amount_by_type"] = save_chart(fig, "avg_amount_by_type.png")

    X = numeric_df[FEATURE_COLUMNS]
    y = numeric_df["isFraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    metrics = {}
    for name, m in models.items():
        preds = m.predict(X_test)
        confusion = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax,
        )
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        safe_name = name.lower().replace(" ", "_")
        chart_paths[f"confusion_matrix_{safe_name}"] = save_chart(
            fig, f"confusion_matrix_{safe_name}.png"
        )

        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1_score": f1_score(y_test, preds, zero_division=0),
            "confusion_matrix": confusion.tolist(),
        }

    stats = {
        "sample_size": len(df),
        "fraud_transactions": int(df["isFraud"].sum()),
        "legitimate_transactions": int((df["isFraud"] == 0).sum()),
        "fraud_rate": float(df["isFraud"].mean() * 100),
        "average_amount": float(df["amount"].mean()),
        "maximum_amount": float(df["amount"].max()),
        "transaction_types_count": int(df["type"].nunique()),
        "model_accuracy": float(metrics["Random Forest"]["accuracy"] * 100),
    }

    insights = [
        f"The dashboard is based on a working sample of {stats['sample_size']:,} transactions for fast analysis.",
        f"Fraud cases make up {stats['fraud_rate']:.3f}% of the sampled transactions, showing a highly imbalanced problem.",
        f"Average transaction value in the sample is ₹{stats['average_amount']:,.2f}, with large outliers reaching ₹{stats['maximum_amount']:,.2f}.",
        "Transfer and cash-out patterns usually deserve closer monitoring because fraud often clusters around high-risk movement types.",
    ]

    importances = model.feature_importances_
    features_info = []
    feature_meta = {
        "amount": {
            "display": "Transaction Amount",
            "desc": "The total monetary value of the transaction. Large, unusual amounts are a key signature of fraud.",
            "icon": "bi-currency-exchange",
        },
        "oldbalanceOrg": {
            "display": "Sender Initial Balance",
            "desc": "The balance in the sender's account before the transaction. Fraud targets accounts with substantial funds.",
            "icon": "bi-wallet2",
        },
        "newbalanceDest": {
            "display": "Receiver Final Balance",
            "desc": "The destination account balance after the transaction. A sudden influx of funds in inactive destination accounts is highly suspicious.",
            "icon": "bi-bank",
        },
        "oldbalanceDest": {
            "display": "Receiver Initial Balance",
            "desc": "The destination account balance before the transaction. Many fraud destination accounts start with zero balance.",
            "icon": "bi-cash-stack",
        },
        "step": {
            "display": "Time Step",
            "desc": "Represents logical hours of time. Fraud activity often peaks at specific hours or intervals.",
            "icon": "bi-clock",
        },
        "type": {
            "display": "Transaction Type",
            "desc": "The transfer category. Fraud is extremely rare in cash-ins or payments, but common in transfers.",
            "icon": "bi-tag",
        },
        "newbalanceOrig": {
            "display": "Sender Final Balance",
            "desc": "The sender's balance after the transaction. A common fraud indicator is emptying the sender's account completely to 0.",
            "icon": "bi-wallet",
        },
    }

    for col, imp in zip(FEATURE_COLUMNS, importances):
        meta = feature_meta.get(col, {"display": col, "desc": "", "icon": "bi-gear"})
        features_info.append(
            {
                "name": col,
                "display_name": meta["display"],
                "description": meta["desc"],
                "icon": meta["icon"],
                "importance": float(imp),
                "percentage": float(imp * 100),
            }
        )

    features_info = sorted(features_info, key=lambda x: x["importance"], reverse=True)

    return {
        "stats": stats,
        "charts": chart_paths,
        "metrics": metrics,
        "insights": insights,
        "feature_importances": features_info,
    }


APP_CONTENT = build_dashboard_assets()


@app.context_processor
def inject_layout_values():
    return {
        "current_year": datetime.now().year,
        "transaction_types": TRANSACTION_TYPES,
        "type_display_names": TYPE_DISPLAY_NAMES,
        "model_options": ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        "current_user": session.get("username"),
        "current_role": session.get("role"),
        "current_name": session.get("name"),
    }


# ─── Main routes ──────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html", stats=APP_CONTENT["stats"])


@app.route("/dashboard")
def dashboard():
    history = session.get("prediction_history", [])
    return render_template(
        "dashboard.html",
        stats=APP_CONTENT["stats"],
        charts=APP_CONTENT["charts"],
        history=history[-10:][::-1],
    )


@app.route("/analysis")
def analysis():
    return render_template(
        "analysis.html",
        stats=APP_CONTENT["stats"],
        charts=APP_CONTENT["charts"],
        insights=APP_CONTENT["insights"],
    )


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    result = None
    submitted = {
        "step": "",
        "type": "TRANSFER",
        "amount": "",
        "oldbalanceOrg": "",
        "newbalanceOrig": "",
        "oldbalanceDest": "",
        "newbalanceDest": "",
        "model_choice": "Random Forest",
    }

    if request.method == "POST":
        submitted = {
            "step": request.form.get("step", ""),
            "type": request.form.get("type", "TRANSFER"),
            "amount": request.form.get("amount", ""),
            "oldbalanceOrg": request.form.get("oldbalanceOrg", ""),
            "newbalanceOrig": request.form.get("newbalanceOrig", ""),
            "oldbalanceDest": request.form.get("oldbalanceDest", ""),
            "newbalanceDest": request.form.get("newbalanceDest", ""),
            "model_choice": request.form.get("model_choice", "Random Forest"),
        }

        try:
            features = [
                [
                    float(submitted["step"]),
                    float(TYPE_MAPPING[submitted["type"]]),
                    float(submitted["amount"]),
                    float(submitted["oldbalanceOrg"]),
                    float(submitted["newbalanceOrig"]),
                    float(submitted["oldbalanceDest"]),
                    float(submitted["newbalanceDest"]),
                ]
            ]

            chosen_model = models.get(submitted["model_choice"], model)
            prediction = int(chosen_model.predict(features)[0])
            probability = None
            if hasattr(chosen_model, "predict_proba"):
                probability = float(chosen_model.predict_proba(features)[0][1] * 100)

            result = {
                "label": "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction",
                "status": "high-risk" if prediction == 1 else "safe",
                "probability": probability,
                "confidence": (probability if prediction == 1 else 100 - (probability or 0)),
                "model_used": submitted["model_choice"],
            }

            # Save to session history
            history = session.get("prediction_history", [])
            history.append({
                "id": len(history) + 1,
                "type": submitted["type"],
                "amount": submitted["amount"],
                "prediction": "Fraud" if prediction == 1 else "Safe",
                "confidence": f"{result['confidence']:.1f}%",
                "model": submitted["model_choice"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            session["prediction_history"] = history
            session.modified = True

        except (ValueError, KeyError):
            flash("Please enter valid numeric values for all prediction inputs.", "danger")
            return redirect(url_for("predict_page"))

    return render_template("prediction.html", result=result, submitted=submitted)


@app.route("/performance")
def performance():
    return render_template(
        "performance.html",
        metrics=APP_CONTENT["metrics"],
        charts=APP_CONTENT["charts"],
        sample_size=APP_CONTENT["stats"]["sample_size"],
        feature_importances=APP_CONTENT["feature_importances"],
    )


@app.route("/about")
def about():
    return render_template("about.html", stats=APP_CONTENT["stats"])


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "there").strip() or "there"
        flash(f"Thanks {name}, your message has been received! We'll get back to you soon.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")


# ─── New feature routes ───────────────────────────────────────────────────────

@app.route("/upload", methods=["GET", "POST"])
def upload():
    results = None
    filename = None
    error = None

    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or not file.filename.endswith(".csv"):
            error = "Please upload a valid CSV file."
        else:
            try:
                filename = file.filename
                df = pd.read_csv(file)

                # Check required columns; if missing, try to map them
                required = set(FEATURE_COLUMNS)
                available = set(df.columns)
                missing = required - available

                if missing:
                    error = f"CSV is missing columns: {', '.join(missing)}. Required: {', '.join(FEATURE_COLUMNS)}"
                else:
                    df_feat = df[FEATURE_COLUMNS].copy()
                    if df_feat["type"].dtype == object:
                        df_feat["type"] = df_feat["type"].map(TYPE_MAPPING).fillna(0)

                    preds = model.predict(df_feat.values)
                    probs = None
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(df_feat.values)[:, 1] * 100

                    df["Prediction"] = ["Fraud" if p == 1 else "Safe" for p in preds]
                    if probs is not None:
                        df["Confidence_%"] = [f"{p:.1f}" for p in probs]

                    results = df.head(200).to_dict(orient="records")
                    session["upload_results"] = results
                    session["upload_filename"] = filename
                    session.modified = True
                    flash(f"✔ File '{filename}' processed successfully! {int(preds.sum())} fraud transactions found.", "success")
            except Exception as e:
                error = f"Error processing file: {str(e)}"

    return render_template("upload.html", results=results, filename=filename, error=error)


@app.route("/download_upload_csv")
def download_upload_csv():
    results = session.get("upload_results", [])
    if not results:
        flash("No upload results to download.", "warning")
        return redirect(url_for("upload"))

    def generate():
        if results:
            keys = results[0].keys()
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
            yield output.getvalue()

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=fraud_predictions.csv"},
    )


@app.route("/history")
def history():
    prediction_history = session.get("prediction_history", [])
    return render_template("history.html", history=list(reversed(prediction_history)))


@app.route("/history/clear", methods=["POST"])
def clear_history():
    session.pop("prediction_history", None)
    session.modified = True
    flash("Prediction history cleared.", "info")
    return redirect(url_for("history"))


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    tx_type = request.args.get("type", "").strip()
    results = []

    if query or tx_type:
        history = session.get("prediction_history", [])
        for item in history:
            match = True
            if query and str(query).lower() not in str(item.get("id", "")).lower():
                match = False
            if tx_type and item.get("type", "").upper() != tx_type.upper():
                match = False
            if match:
                results.append(item)

    return render_template(
        "search.html",
        results=results,
        query=query,
        tx_type=tx_type,
        transaction_types=TRANSACTION_TYPES,
    )


@app.route("/faq")
def faq():
    return render_template("faq.html")


@app.route("/technologies")
def technologies():
    return render_template("technologies.html")


@app.route("/download_report")
def download_report():
    history = session.get("prediction_history", [])
    if not history:
        flash("No predictions to download yet. Make a prediction first!", "warning")
        return redirect(url_for("predict_page"))

    def generate():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Transaction Type", "Amount", "Prediction", "Confidence", "Model Used", "Date/Time"])
        for item in history:
            writer.writerow([
                item.get("id", ""),
                item.get("type", ""),
                item.get("amount", ""),
                item.get("prediction", ""),
                item.get("confidence", ""),
                item.get("model", ""),
                item.get("timestamp", ""),
            ])
        yield output.getvalue()

    return Response(
        generate(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment;filename=fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        },
    )


# ─── Auth routes ──────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        action = request.form.get("action", "login")

        if action == "register":
            username = request.form.get("reg_username", "").strip()
            password = request.form.get("reg_password", "").strip()
            name = request.form.get("reg_name", "").strip()

            if not username or not password or not name:
                flash("All registration fields are required.", "danger")
            elif username in USERS:
                flash("Username already exists. Please choose a different one.", "danger")
            elif len(password) < 6:
                flash("Password must be at least 6 characters.", "danger")
            else:
                USERS[username] = {"password": password, "role": "user", "name": name}
                flash(f"Account created for {name}! You can now log in.", "success")
                return redirect(url_for("login"))

        else:  # login
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()
            user = USERS.get(username)

            if user and user["password"] == password:
                session["username"] = username
                session["role"] = user["role"]
                session["name"] = user["name"]
                flash(f"Welcome back, {user['name']}! 🎉", "success")
                return redirect(url_for("home"))
            else:
                flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    name = session.get("name", "User")
    session.clear()
    flash(f"Goodbye {name}, you have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/admin")
@admin_required
def admin():
    all_history = session.get("prediction_history", [])
    upload_results = session.get("upload_results", [])
    upload_filename = session.get("upload_filename", None)
    return render_template(
        "admin.html",
        all_history=list(reversed(all_history)),
        upload_results=upload_results[:50],
        upload_filename=upload_filename,
        users=list(USERS.keys()),
        stats=APP_CONTENT["stats"],
    )


# ─── API routes ───────────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    history = session.get("prediction_history", [])
    fraud_count = sum(1 for h in history if h.get("prediction") == "Fraud")
    safe_count = sum(1 for h in history if h.get("prediction") == "Safe")
    return jsonify({
        "total_transactions": APP_CONTENT["stats"]["sample_size"],
        "fraud_transactions": APP_CONTENT["stats"]["fraud_transactions"],
        "legitimate_transactions": APP_CONTENT["stats"]["legitimate_transactions"],
        "fraud_rate": round(APP_CONTENT["stats"]["fraud_rate"], 3),
        "model_accuracy": round(APP_CONTENT["stats"]["model_accuracy"], 2),
        "session_predictions": len(history),
        "session_fraud": fraud_count,
        "session_safe": safe_count,
        "timestamp": datetime.now().isoformat(),
    })


if __name__ == "__main__":
    app.run(debug=True)
