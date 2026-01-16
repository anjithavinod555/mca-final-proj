# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from myapp import feature_extraction
#
#
# def getResult_rand(url):
#
#     print(url,"-----------------------------------------")
#     #Importing dataset
#     data = pd.read_csv('C:\\Users\\amaya\\PycharmProjects\\phishing_det\\myapp\\det\\phishing.csv',delimiter=",")
#
#     #Seperating features and labels
#     X = np.array(data.iloc[: , :-1])
#     y = np.array(data.iloc[: , -1])
#
#     print(type(X))
#     #Seperating training features, testing features, training labels & testing labels
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#    # classifier = RandomForestClassifier()
#     classifier = LogisticRegression()
#     classifier.fit(X_train, y_train)
#     # score = classifier.score(X_test, y_test)
#     # score = score*100
#     # print(score,"::::::::::::score")
#
#     X_new = []
#
#     X_input = url
#     X_new=feature_extraction.generate_data_set(X_input)
#     while len(X_new) < 30:
#         X_new.append(1)
#     X_new = np.array(X_new).reshape(1,-1)
#
#     analysis_result = ""
#
#     try:
#         prediction = classifier.predict(X_new)
#         print(prediction)
#         if prediction == -1:
#             analysis_result = "Phishing URL"
#         elif prediction == 0:
#             analysis_result = "This website has been detected as Suspecious"
#         else:
#             analysis_result = "This website has been detected as Legitimate URL"
#     except Exception as a:
#         print(a)
#         analysis_result = "This website has been detected as Phishing URL"
#
#     print(analysis_result)
#     return analysis_result
#
# getResult_rand("https://www.amazon.com/")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from myapp import feature_extraction


def getResult_rand(url):

    print(url, "-----------------------------------------")

    # Importing dataset
    data = pd.read_csv(
        'C:\\Users\\amaya\\PycharmProjects\\phishing_det\\myapp\\det\\phishing.csv',
        delimiter=","
    )

    # Separating features and labels
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Random Forest Classifier
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    classifier.fit(X_train, y_train)

    # Feature extraction for new URL
    X_new = feature_extraction.generate_data_set(url)

    # Ensure feature length = 30
    while len(X_new) < 30:
        X_new.append(1)

    X_new = np.array(X_new).reshape(1, -1)

    analysis_result = ""

    try:
        prediction = classifier.predict(X_new)[0]
        print("Prediction:", prediction)

        if prediction == -1:
            analysis_result = "Phishing URL"
        elif prediction == 0:
            analysis_result = "This website has been detected as Suspicious"
        else:
            analysis_result = "This website has been detected as Legitimate URL"

    except Exception as e:
        print(e)
        analysis_result = "This website has been detected as Phishing URL"

    print(analysis_result)
    return analysis_result


# Test
getResult_rand("https://www.amazon.com/")


#====================================================================================

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt
# from myapp import feature_extraction
#
#
# def getResult(url):
#     print(url, "-----------------------------------------")
#
#     # Importing dataset
#     data = pd.read_csv(
#         'C:\\Users\\dhars\\PycharmProjects\\CyberSecurity\\myapp\\dataset\\dataset.csv',
#         delimiter=",")
#
#     # Separating features and labels
#     X = np.array(data.iloc[:, :-1])
#     y = np.array(data.iloc[:, -1])
#
#     print(type(X))
#
#     # Splitting dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Logistic Regression classifier
#     classifier = LogisticRegression(max_iter=1000)
#     classifier.fit(X_train, y_train)
#
#     # Model score
#     score = classifier.score(X_test, y_test) * 100
#     print(score, ":::::::::::: score")
#
#     # Predict on test data for confusion matrix
#     y_pred = classifier.predict(X_test)
#
#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:")
#     print(cm)
#
#     # Plot confusion matrix
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Phishing', 'Suspicious', 'Legitimate'],
#                 yticklabels=['Phishing', 'Suspicious', 'Legitimate'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.show()
#
#     # Classification Report
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))
#
#     # Predict a new URL
#     X_input = url
#     X_new = feature_extraction.generate_data_set(X_input)
#
#     # Padding in case the feature vector is shorter than expected
#     # while len(X_new) < 30:
#     #     X_new.append(1)
#
#     X_new = np.array(X_new).reshape(1, -1)
#
#     # Analyze prediction
#     analysis_result = ""
#     try:
#         prediction = classifier.predict(X_new)
#         print(prediction)
#
#         if prediction == -1:
#             analysis_result = "Phishing URL"
#         elif prediction == 0:
#             analysis_result = "This website has been detected as Suspicious"
#         else:
#             analysis_result = "This website has been detected as Legitimate URL"
#     except Exception as e:
#         print(e)
#         analysis_result = "This website has been detected as Phishing URL"
#
#     print(analysis_result)
#     return analysis_result


# Example usage:
# getResult(" https://testsafebrowsing.appspot.com/")


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt
# from myapp import feature_extraction
#
#
# def getResult(url):
#     print(url, "-----------------------------------------")
#
#     # Importing dataset
#     data = pd.read_csv(
#         'C:\\Users\\dhars\\PycharmProjects\\CyberSecurity\\myapp\\dataset\\dataset.csv',
#         delimiter=",")
#
#     # Separating features and labels
#     X = np.array(data.iloc[:, :-1])
#     y = np.array(data.iloc[:, -1])
#
#     print(type(X))
#
#     # Splitting dataset into training and testing sets
#     X_train, X_test,X_temp, y_train, y_test,y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#
#
#     # Logistic Regression classifier
#     classifier = LogisticRegression(max_iter=1000)
#     classifier.fit(X_train, y_train)
#
#     # Training accuracy
#     train_score = classifier.score(X_train, y_train) * 100
#     print(f"Training Accuracy: {train_score:.2f}%")
#
#     val_score = classifier.score(X_val, y_val) * 100
#     print(f"Validation Accuracy: {val_score:.2f}%")
#
#     # Testing accuracy
#     test_score = classifier.score(X_test, y_test) * 100
#     print(f"Testing Accuracy: {test_score:.2f}% :::::::::::: score")
#
#     # Predict on test data for confusion matrix
#     y_pred = classifier.predict(X_test)
#
#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:")
#     print(cm)
#
#     # Plot confusion matrix
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Phishing', 'Suspicious', 'Legitimate'],
#                 yticklabels=['Phishing', 'Suspicious', 'Legitimate'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.show()
#
#     # Classification Report
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))
#
#     # Predict a new URL
#     X_input = url
#     X_new = feature_extraction.generate_data_set(X_input)
#
#     X_new = np.array(X_new).reshape(1, -1)
#
#     # Analyze prediction
#     analysis_result = ""
#     try:
#         prediction = classifier.predict(X_new)
#         print(prediction)
#
#         if prediction == -1:
#             analysis_result = "Phishing URL"
#         elif prediction == 0:
#             analysis_result = "This website has been detected as Suspicious"
#         else:
#             analysis_result = "This website has been detected as Legitimate URL"
#     except Exception as e:
#         print(e)
#         analysis_result = "This website has been detected as Phishing URL"
#
#     print(analysis_result)
#     return analysis_result


# getResult("https://vxvault.net/ViriList.php")



# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt
# from myapp import feature_extraction
#
#
# def getResult(url):
#     print(url, "-----------------------------------------")
#
#     # Importing dataset
#     data = pd.read_csv(
#         'C:\\Users\\dhars\\PycharmProjects\\CyberSecurity\\myapp\\dataset\\dataset.csv',
#         delimiter=",")
#
#     # Separating features and labels
#     X = np.array(data.iloc[:, :-1])
#     y = np.array(data.iloc[:, -1])
#
#     print(type(X))
#
#     # First split: train + temp (60% train, 40% temp)
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
#
#     # Second split: validation + test (50% of temp each = 20% total for each)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#
#     # Logistic Regression classifier
#     classifier = LogisticRegression(max_iter=1000)
#     classifier.fit(X_train, y_train)
#
#     # Training accuracy
#     train_score = classifier.score(X_train, y_train) * 100
#     print(f"Training Accuracy: {train_score:.2f}%")
#
#     # Validation accuracy
#     val_score = classifier.score(X_val, y_val) * 100
#     print(f"Validation Accuracy: {val_score:.2f}%")
#
#     # Testing accuracy
#     test_score = classifier.score(X_test, y_test) * 100
#     print(f"Testing Accuracy: {test_score:.2f}% :::::::::::: score")
#
#     # Predict on test data for confusion matrix
#     y_pred = classifier.predict(X_test)
#
#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:")
#     print(cm)
#
#     # Plot confusion matrix
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Phishing', 'Suspicious', 'Legitimate'],
#                 yticklabels=['Phishing', 'Suspicious', 'Legitimate'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.show()
#
#     # Classification Report
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))
#
#     # Predict a new URL
#     X_input = url
#     X_new = feature_extraction.generate_data_set(X_input)
#     X_new = np.array(X_new).reshape(1, -1)
#
#     # Analyze prediction
#     analysis_result = ""
#     try:
#         prediction = classifier.predict(X_new)
#         print(prediction)
#
#         if prediction == -1:
#             analysis_result = "Phishing URL"
#         elif prediction == 0:
#             analysis_result = "This website has been detected as Suspicious"
#         else:
#             analysis_result = "This website has been detected as Legitimate URL"
#     except Exception as e:
#         print(e)
#         analysis_result = "This website has been detected as Phishing URL"
#
#     print(analysis_result)
#     return analysis_result
#
#
# # Run the function
# getResult("https://vxvault.net/ViriList.php")


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt
# from myapp import feature_extraction


# def getResult(url):
#     print(url, "-----------------------------------------")
#
#     # Importing dataset
#     data = pd.read_csv(
#         'C:\\Users\\dhars\\PycharmProjects\\CyberSecurity\\myapp\\dataset\\dataset.csv',
#         delimiter=",")
#
#     # Separating features and labels
#     X = np.array(data.iloc[:, :-1])
#     y = np.array(data.iloc[:, -1])
#
#     print(type(X))
#
#     # First split: train + temp (60% train, 40% temp)
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
#
#     # Second split: validation + test (50% of temp each = 20% total for each)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#
#     # Logistic Regression classifier
#     classifier = LogisticRegression(max_iter=1000)
#     classifier.fit(X_train, y_train)
#
#     # Training accuracy
#     train_score = classifier.score(X_train, y_train) * 100
#     print(f"Training Accuracy: {train_score:.2f}%")
#
#     # Validation accuracy
#     val_score = classifier.score(X_val, y_val) * 100
#     print(f"Validation Accuracy: {val_score:.2f}%")
#
#     # Testing accuracy
#     test_score = classifier.score(X_test, y_test) * 100
#     print(f"Testing Accuracy: {test_score:.2f}% :::::::::::: score")
#
#     # Predict on test data for confusion matrix
#     y_pred = classifier.predict(X_test)
#
#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:")
#     print(cm)
#
#     # Plot confusion matrix
#     # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#     #             xticklabels=['Phishing', 'Suspicious', 'Legitimate'],
#     #             yticklabels=['Phishing', 'Suspicious', 'Legitimate'])
#     # plt.xlabel('Predicted')
#     # plt.ylabel('Actual')
#     # plt.title('Confusion Matrix')
#
#     #================================
#     # plt.show()
#
#     #========================
#
#     # Classification Report
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))
#
#     # Predict a new URL
#     X_input = url
#     X_new = feature_extraction.generate_data_set(X_input)
#     X_new = np.array(X_new).reshape(1, -1)
#     print(X_new)
#     print("mmmmmmmmmmmmmmmmmmmmmmmm")
#
#     # Analyze prediction
#     analysis_result = ""
#     try:
#         prediction = classifier.predict(X_new)
#
#         print(prediction)
#
#         if prediction == -1:
#             analysis_result = "Phishing URL"
#         elif prediction == 0:
#             analysis_result = "This website has been detected as Suspicious"
#         else:
#             analysis_result = "This website has been detected as Legitimate URL"
#     except Exception as e:
#         print(e)
#         analysis_result = "This website has been detected as Phishing URL"
#
#     print(analysis_result)
#     return analysis_result


# Run the function
# getResult("https://vxvault.net/ViriList.php")
