import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_pipeline.pkl"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Force-clean target column
    df["Churn"] = df["Churn"].astype(str).str.strip()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def main():
    os.makedirs("models", exist_ok=True)

    df = load_data(DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved model pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    main()