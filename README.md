# 🛡️ Smart Fraud Detection System

A Machine Learning-based web application designed to detect fraudulent financial transactions. The system uses Python, Flask, and Scikit-learn to analyze transaction data and predict whether a transaction is fraudulent or legitimate.

---

## 🚀 Features

- 🤖 Machine Learning-based Fraud Prediction
- 📊 Interactive Dashboard
- 📈 Data Analysis and Visualizations
- 🔍 Transaction Search
- 📜 Prediction History
- 📂 Dataset Upload Support
- 📉 Model Performance Analysis
- 🔐 User Login Interface
- 📱 Responsive Web Design

---

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- HTML
- CSS

---

## 📂 Project Structure

```text
Smart-Fraud-Detection-System/
│
├── models/                 # Trained Machine Learning models
├── static/                 # CSS, images, and static files
├── templates/              # HTML web pages
├── visualizations/         # Data visualization scripts
│
├── app.py                  # Flask application
├── train_model.py          # Model training script
├── requirements.txt        # Required Python libraries
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The original dataset is not included in this repository because of its large file size.

To retrain the Machine Learning models, download the dataset and place it inside the following folder:

```text
data/
└── PS_20174392719_1491204439457_log.csv
```

> **Note:** The trained models are included in the `models/` folder. Depending on the application configuration, the Flask application may run without retraining the models.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/satya12satya/Smart-Fraud-Detection-System.git
```

### 2. Navigate to the Project Folder

```bash
cd Smart-Fraud-Detection-System
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

Then open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🔄 Project Workflow

```text
User
   ↓
Web Application
   ↓
Flask Backend
   ↓
Machine Learning Model
   ↓
Fraud Prediction
   ↓
Dashboard & Analytics
```

---

## 📊 Visualizations

The project includes data analysis and visualization features such as:

- Fraud vs Non-Fraud Distribution
- Transaction Type Analysis
- Transaction Amount Analysis
- Model Performance Analysis
- Confusion Matrix
- Transaction Analytics

---

## 🤖 Machine Learning Models

The trained Machine Learning models are stored in the `models/` directory.

The system analyzes transaction-related information and predicts whether a transaction is:

- 🚨 **Fraudulent**
- ✅ **Legitimate**

The `train_model.py` script can be used to train or retrain the Machine Learning models when the dataset is available.

---

## 📁 Important Files

| File/Folder | Description |
|---|---|
| `app.py` | Main Flask web application |
| `train_model.py` | Machine Learning model training script |
| `models/` | Saved trained ML models |
| `templates/` | HTML pages |
| `static/` | CSS, images, and other static resources |
| `visualizations/` | Visualization-related scripts |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## 🔮 Future Enhancements

- Real-time fraud monitoring
- Database integration
- Advanced user authentication
- CSV batch prediction
- Downloadable prediction reports
- Cloud deployment
- Real-time notifications
- Improved Machine Learning models
- API integration

---

## 👩‍💻 Author

**Eguduru Pushpa**

---

## ⭐ Support

If you found this project useful, please consider giving the repository a ⭐ star!

---

### 📌 Project Purpose

This project was developed to demonstrate the practical application of **Machine Learning, Python, Flask, and data visualization** for identifying potentially fraudulent financial transactions.
