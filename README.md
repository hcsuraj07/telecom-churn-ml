# 🚀 Telecom Customer Churn Prediction  
### End-to-End Machine Learning Pipeline with FastAPI & Docker

An end-to-end machine learning project that predicts customer churn using telecom customer data. This project demonstrates the complete lifecycle of an ML system — from data preprocessing and model training to deployment using FastAPI and Docker.

---

## 📌 Project Overview

Customer churn is a critical problem for subscription-based businesses. This project focuses on identifying customers who are likely to leave, enabling businesses to take proactive retention actions.

The solution includes:
- Data preprocessing and cleaning  
- Feature engineering and exploratory analysis  
- Model training and evaluation  
- Model explainability using SHAP  
- API deployment with FastAPI  
- Containerization with Docker  

---

## 🧠 Business Problem

Telecom companies face revenue loss due to customer churn. Predicting churn helps:
- Identify at-risk customers early  
- Improve retention strategies  
- Optimize customer engagement  

---

## 📊 Dataset

- **Source:** Telco Customer Churn Dataset  
- **Size:** ~7,000 customers  
- **Features include:**
  - Demographics (gender, senior citizen)
  - Account details (tenure, contract type)
  - Services (internet, streaming, support)
  - Billing (monthly & total charges)

- **Target:** Churn (Yes/No)

---

## 🧹 Data Preparation

- Dropped non-informative column (`customerID`)  
- Cleaned inconsistent values in `TotalCharges`  
- Handled missing values  
- Encoded categorical variables  
- Ensured correct data types for modeling  

---

## 🔍 Exploratory Data Analysis

Key insights:
- Customers with **short tenure** are more likely to churn  
- **Month-to-month contracts** show higher churn  
- **Higher monthly charges** slightly increase churn risk  

---

## 📈 Feature Insights (Correlation)

| Feature | Correlation with Churn |
|--------|------------------------|
| tenure | -0.35 |
| MonthlyCharges | 0.19 |
| TotalCharges | -0.20 |

👉 Tenure is the strongest predictor of churn.

---

## 🤖 Model Development

Models trained:
- Logistic Regression  
- Random Forest  
- XGBoost (final model)

### Evaluation Metrics:
- Accuracy  
- Precision  
- Recall (critical for churn detection)  
- F1-score  
- ROC-AUC  

---

## 📊 Model Performance

- **Accuracy:** 0.80  
- **ROC-AUC:** 0.8433  
- **Recall (Churn):** 0.53  

---

## 🔍 Explainable AI (SHAP)

Used SHAP to interpret model predictions.

Key findings:
- Tenure has the strongest impact  
- Contract type significantly affects churn  
- Monthly charges influence customer decisions  

---

## 🧱 Architecture

```
Data → Cleaning → Feature Engineering → Model Training → Evaluation → API → Docker
```

---

## ⚙️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- SHAP  
- FastAPI  
- Uvicorn  
- Docker  
- Joblib  

---

## 📂 Project Structure

```
telecom-churn-ml/
├── app/
│   └── main.py
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   └── churn_pipeline.pkl
├── src/
│   ├── train.py
│   └── explain.py
├── notebooks/
│   └── eda.ipynb
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
```

---

## 🚀 Run Locally

### 1. Create environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train model
```bash
python src/train.py
```

### 4. Run API
```bash
uvicorn app.main:app --reload
```

### 5. Open Swagger UI
```
http://127.0.0.1:8000/docs
```

---

## 🐳 Run with Docker

### Build image
```bash
docker build -t telecom-churn-ml .
```

### Run container
```bash
docker run -p 8000:8000 telecom-churn-ml
```

### Access API
```
http://127.0.0.1:8000/docs
```

---

## 🔌 API Example

### Endpoint
`POST /predict`

### Input
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.85,
  "TotalCharges": 1089.0
}
```

### Output
```json
{
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.7724
}
```

---

## 💼 Key Learnings

- Handling real-world messy data  
- Feature engineering and model evaluation  
- Importance of recall in classification  
- Model explainability with SHAP  
- Building APIs using FastAPI  
- Deploying ML models using Docker  

---

## 🔥 Future Improvements

- Deploy API to cloud (AWS / Render)  
- Add frontend UI  
- Improve model recall with tuning  
- Add real-time prediction pipeline  

---

## 👤 Author

**Suraj Chandra**  
Machine Learning & Data Analytics Enthusiast  

GitHub: https://github.com/hcsuraj07
