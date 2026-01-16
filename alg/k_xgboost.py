import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier

# =========================
# Load Dataset
# =========================
data = pd.read_csv(r"C:\Riss\phishing_det\myapp\alg\phishing.csv")
data = data.drop(['Index'], axis=1)

# 🔥 FIX: Convert labels
data['class'] = data['class'].map({-1: 0, 1: 1})

# =========================
# Split Features & Label
# =========================
X = data.drop("class", axis=1)
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# XGBoost Model
# =========================
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = xgb_model.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# =========================
# Save Model
# =========================
with open("phishing_xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("✅ Model saved successfully")
