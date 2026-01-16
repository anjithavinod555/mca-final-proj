import pickle
import smtplib

from django.contrib import messages
from django.contrib.auth import logout, authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User, Group
from django.shortcuts import render, redirect
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.calibration import label_binarize
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split

# Create your views here.
from myapp.feature import FeatureExtraction
from myapp.models import Register


def login_get(request):
    return render(request,'login_index.html')

def login_post(request):
    username=request.POST['u1']
    password=request.POST['p1']
    print(request.POST,"=============")
    log=authenticate(username=username,password=password)
    # print("hhhhhhhh===========")
    if log is not None:
        login(request,log)
        print("============aaaaaaaaaaaaa")
        if log.groups.filter(name='users'):
            return redirect('/myapp/home_get/')
        else:
            messages.warning(request,'User Not Found!')
            return redirect('/myapp/login_get/')
    else:
        messages.warning(request,'User Not Found!')
        return redirect('/myapp/login_get/')


def signup_get(request):
    return render(request,'signup_index.html')

def signup_post(request):
    name=request.POST['fname']
    # username=request.POST['uname']
    email=request.POST['mail']
    number=request.POST['pnumber']
    password=request.POST['pass']
    confirm_password=request.POST['conpass']
    print(request.POST)

    if User.objects.filter(username=email).exists():
        messages.error(request,'Email already exist!!')
        return redirect('/myapp/signup_get/')

    elif password==confirm_password:
        user=User.objects.create_user(username=email,password=confirm_password)
        user.groups.add(Group.objects.get(name='users'))
        user.save()

        r=Register()
        r.name=name
        r.username=""
        r.mail=email
        r.phone=number
        r.USER=user
        r.save()

        messages.success(request,'Registered Successfully!....')
        return redirect('/myapp/login_get/')
    else:
        messages.warning(request,'Invalid Credentials')
        return redirect('/myapp/signup_get/')

@login_required(login_url='/myapp/loginpage_get/')
def home_get(request):
    return render(request,'home_index.html')

@login_required(login_url='/myapp/loginpage_get/')
def view_profile(request):
    data=Register.objects.get(USER_id=request.user.id)
    return render(request,'view_profile.html',{'data':data})

@login_required(login_url='/myapp/loginpage_get/')
def change_get(request):
    return render(request,'change_password.html')

@login_required(login_url='/myapp/loginpage_get/')
def change_post(request):
    current=request.POST['p1']
    new=request.POST['p2']
    confirm=request.POST['p3']
    user=request.user
    if user.check_password(current):
        if new==confirm:
            user.set_password(new)
            user.save()
            return redirect('/myapp/login_get/')
        else:
            messages.warning(request,'Password Does Not Match!')
            return redirect('/myapp/change_get/')
    else:
        messages.warning(request,'Invalid Credentials')
        return redirect('/myapp/change_get/')

def forgot_get(request):
    return render(request,'forgot.html')

def forgot_post(request):
    username=request.POST['uu1']
    print(request.POST)
    # res = User.objects.filter(username=username)
    user=User.objects.get(username=username)
    if user is not None:
        import random
        new_pass = random.randint(000000,999999)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login("mysql695@gmail.com", "ghcy kfxl zmfb lirr")  # App Password
        to = username
        subject = "Test Email"
        body = "Your new password is " + str(new_pass)
        msg = f"Subject: {subject}\n\n{body}"
        server.sendmail("s@gmail.com", to, msg)
        # Disconnect from the server
        server.quit()
        user.set_password(new_pass)
        user.save()
        messages.success(request,'Password Updated Successfully! Please Check Your Email.')
        return redirect('/myapp/login_get/')
    else:
        messages.warning(request,'User Not Found')
        return redirect('/myapp/forgot_get/')




def logout_page(request):
    logout(request)
    return redirect('/myapp/login_get/')

@login_required(login_url='/myapp/loginpage_get/')
def upload_new_get(request):
    return render(request,'upload_new.html')



@login_required(login_url='/myapp/loginpage_get/')
def upload_new_post(request):
    url = request.POST["url"]
    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1,30) 
    file = open("/Users/sreeshyam/Desktop/phishing_det/phishing_random_model.pkl","rb")
    gbc = pickle.load(file)
    file.close()

    y_pred =gbc.predict(x)[0]
    #1 is safe       
    #-1 is unsafe
    y_pro_phishing = gbc.predict_proba(x)[0,0]
    y_pro_non_phishing = gbc.predict_proba(x)[0,1]
    # if(y_pred ==1 ):
    pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
    return render(request,'upload_new.html',{ 'xx': round(y_pro_non_phishing,2),'url':url })



def random_visualization(request):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 confusion_matrix, roc_curve, auc, classification_report)

    data = pd.read_csv(r"/Users/sreeshyam/Desktop/phishing_det/myapp/alg/phishing.csv")
    data = data.drop(['Index'], axis=1)

    X = data.drop("class", axis=1)
    y = data["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with open("/Users/sreeshyam/Desktop/phishing_det/phishing_random_model.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC values
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Classification report as dict
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics_list = [
    {"name": "Accuracy", "value": round(accuracy * 100, 2)},
    {"name": "Precision", "value": round(precision * 100, 2)},
    {"name": "Recall", "value": round(recall * 100, 2)},
    {"name": "F1 Score", "value": round(f1 * 100, 2)},
    ]      

    context = {
        "metrics": metrics_list,
        "cm": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": round(roc_auc, 3),
        "report": report
    }


    return render(request, "random_performance.html", context)

 


@login_required(login_url='/myapp/loginpage_get/')
def upload_xg_get(request):
    return render(request,'upload_xg.html')



@login_required(login_url='/myapp/loginpage_get/')
def upload_xg_post(request):
    url = request.POST["url"]
    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1,30) 
    file = open("/Users/sreeshyam/Desktop/phishing_det/phishing_xgboost_model.pkl","rb")
    gbc = pickle.load(file)
    file.close()

    y_pred =gbc.predict(x)[0]
    #1 is safe       
    #-1 is unsafe
    y_pro_phishing = gbc.predict_proba(x)[0,0]
    y_pro_non_phishing = gbc.predict_proba(x)[0,1]
    # if(y_pred ==1 ):
    pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
    return render(request,'upload_xg.html',{ 'xx': round(y_pro_non_phishing,2),'url':url })


@login_required(login_url='/myapp/loginpage_get/')
def upload_svm_get(request):
    return render(request,'upload_svm.html')



@login_required(login_url='/myapp/loginpage_get/')
def upload_svm_post(request):
    url = request.POST["url"]
    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1,30) 
    file = open(r"/Users/sreeshyam/Desktop/phishing_det/phishing_svm_model.pkl","rb")
    gbc = pickle.load(file)
    file.close()

    y_pred =gbc.predict(x)[0]
    #1 is safe       
    #-1 is unsafe
    y_pro_phishing = gbc.predict_proba(x)[0,0]
    y_pro_non_phishing = gbc.predict_proba(x)[0,1]
    # if(y_pred ==1 ):
    pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
    return render(request,'upload_svm.html',{ 'xx': round(y_pro_non_phishing,2),'url':url })



def svm_visualization(request):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 confusion_matrix, roc_curve, auc, classification_report)

    data = pd.read_csv("/Users/sreeshyam/Desktop/phishing_det/myapp/alg/phishing.csv")
    data = data.drop(['Index'], axis=1)

    X = data.drop("class", axis=1)
    y = data["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with open("/Users/sreeshyam/Desktop/phishing_det/phishing_svm_model.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC values
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Classification report as dict
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics_list = [
    {"name": "Accuracy", "value": round(accuracy * 100, 2)},
    {"name": "Precision", "value": round(precision * 100, 2)},
    {"name": "Recall", "value": round(recall * 100, 2)},
    {"name": "F1 Score", "value": round(f1 * 100, 2)},
    ]      

    context = {
        "metrics": metrics_list,
        "cm": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": round(roc_auc, 3),
        "report": report
    }


    return render(request, "random_svm.html", context)

 
 

def xg_visualization(request):
    import pandas as pd

    # ---------------- LOAD DATA ----------------
    data = pd.read_csv(r"/Users/sreeshyam/Desktop/phishing_det/myapp/alg/phishing.csv")
    data = data.drop(['Index'], axis=1)

    X = data.drop("class", axis=1)
    y = data["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- LOAD PICKLE MODEL ----------------
    with open("/Users/sreeshyam/Desktop/phishing_det/phishing_xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)

    # ---------------- PREDICTIONS ----------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)   # multiclass probabilities

    # ---------------- METRICS (MULTICLASS SAFE) ----------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y_test, y_pred)

    # ---------------- MULTICLASS ROC (MICRO AVERAGE) ----------------
    classes = sorted(y.unique())
    y_test_bin = label_binarize(y_test, classes=classes)

    fpr, tpr, _ = roc_curve(
        y_test_bin.ravel(),
        y_pred.ravel()
    )
    roc_auc = auc(fpr, tpr)

    # ---------------- CLASSIFICATION REPORT ----------------
    raw_report = classification_report(y_test, y_pred, output_dict=True)

    # clean keys for Django templates
    clean_report = {}
    for label, values in raw_report.items():
        if isinstance(values, dict):
            clean_report[label] = {
                "precision": values.get("precision", 0),
                "recall": values.get("recall", 0),
                "f1": values.get("f1-score", 0),
                "support": values.get("support", 0),
            }

    # ---------------- METRICS LIST (FOR BOOTSTRAP UI) ----------------
    metrics_list = [
        {"name": "Accuracy", "value": round(accuracy * 100, 2)},
        {"name": "Precision", "value": round(precision * 100, 2)},
        {"name": "Recall", "value": round(recall * 100, 2)},
        {"name": "F1 Score", "value": round(f1 * 100, 2)},
    ]

    # ---------------- CONTEXT ----------------
    context = {
        "metrics": metrics_list,
        "cm": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": round(roc_auc, 3),
        "report": clean_report,
        "classes": classes
    }

    # return render(request, "xgboost_performance.html", context)



    return render(request, "xgboost_performance.html", context)

 
 


@login_required(login_url='/myapp/loginpage_get/')
def upload_transformerbased(request):

    
    return render(request,'upload_svm.html')



@login_required(login_url='/myapp/loginpage_get/')
def upload_transformerbased_post(request):
    url = request.POST["url"]
    
    return render(request,'upload_svm.html')
