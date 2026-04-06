# Telecom Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn using telecom customer data. The project includes data preprocessing, model training with XGBoost, and deployment through a FastAPI prediction API.

## Overview
This project helps identify customers at risk of leaving a telecom service. It uses structured customer account, service, and billing data to train a classification model and expose predictions through a REST API.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- XGBoost
- FastAPI
- Joblib

## Project Structure
```text
telecom-churn-ml/
├── app/
│   └── main.py
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
├── src/
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
