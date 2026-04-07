import joblib
import pandas as pd
import shap

from sklearn.model_selection import train_test_split

DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_pipeline.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.drop("customerID", axis=1)

    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = df.dropna()

    return df


df = load_data()

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = joblib.load(MODEL_PATH)

# Extract trained model
model = pipeline.named_steps["model"]

# Transform data using pipeline preprocessor
X_processed = pipeline.named_steps["preprocessor"].transform(X_test)

# SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_processed)

# Plot
shap.summary_plot(shap_values, X_processed)