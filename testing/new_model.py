import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# ============================
# LOAD DATA
# ============================

df = pd.read_csv("heart_disease_enriched.csv")

# ============================
# FEATURES & TARGET
# ============================

X = df.drop("heart_disease", axis=1)
y = df["heart_disease"]

# ============================
# COLUMN GROUPS
# ============================

categorical_cols = [
    "physical_activity_level",
    "alcohol_intake"
]

numeric_cols = [
    "gender",
    "age",
    "hypertension",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes",
    "total_cholesterol",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "family_history"
]

# ============================
# PREPROCESSING
# ============================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ============================
# MODEL (SENSITIVE TO EXTREMES)
# ============================

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=5,
    class_weight={0: 1, 1: 1.5},
    random_state=42,
    n_jobs=-1
)

# ============================
# PIPELINE
# ============================

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", rf_model)
])

# ============================
# TRAIN / TEST SPLIT
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

# ============================
# EVALUATION
# ============================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# ============================
# SAVE MODEL
# ============================

joblib.dump(pipeline, "heart_disease_pipeline.pkl")
print("Pipeline saved successfully")
