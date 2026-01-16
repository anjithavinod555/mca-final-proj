import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r"C:\Riss\phishing_det\myapp\alg\phishing.csv")
data.head()

data = data.drop(['Index'],axis = 1)


df = data[['PrefixSuffix-', 'SubDomains', 'HTTPS','AnchorURL','WebsiteTraffic','class']]


data['class'].value_counts().plot(kind='pie',autopct='%1.2f%%')

X = data.drop(["class"],axis =1)
y = data["class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


from sklearn.ensemble import RandomForestClassifier
#from sklearn.pipeline import Pipeline

# instantiate the model
log = RandomForestClassifier()

# fit the model 
log.fit(X_train,y_train)

y_train_log = log.predict(X_train)
y_test_log = log.predict(X_test)


print(metrics.classification_report(y_test, y_test_log))

# Save the trained model as a pickle file
with open("phishing_logistic_model.pkl", "wb") as file:
    pickle.dump(log, file)

print("Model saved successfully as phishing_logistic_model.pkl")




