import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# =========================
# Load Dataset
# =========================
data = pd.read_csv(r"C:\Riss\phishing_det\myapp\alg\phishing.csv")
data = data.drop(['Index'], axis=1)

df = data[['PrefixSuffix-', 'SubDomains', 'HTTPS',
           'AnchorURL', 'WebsiteTraffic', 'class']]

# Class distribution
data['class'].value_counts().plot(
    kind='pie', autopct='%1.2f%%', title="Class Distribution"
)
plt.show()

# =========================
# Feature & Label Split
# =========================
X = data.drop(["class"], axis=1)
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# =========================
# SVM Model
# =========================
svm_model = SVC(
    kernel='rbf',        # rbf works well for phishing data
    C=10,
    gamma='scale',
    probability=True,   # needed if you want probabilities later
    random_state=42
)

# Train model
svm_model.fit(X_train, y_train)

# Predictions
y_test_pred = svm_model.predict(X_test)

# =========================
# Evaluation
# =========================
print(metrics.classification_report(y_test, y_test_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_test_pred))

# =========================
# Save Model
# =========================
with open("phishing_svm_model.pkl", "wb") as file:
    pickle.dump(svm_model, file)

print("✅ Model saved successfully as phishing_svm_model.pkl")
