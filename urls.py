"""phishing_det URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from myapp import views

urlpatterns = [
    path('login_get/', views.login_get),
    path('signup_get/', views.signup_get),
    path('signup_post/', views.signup_post),
    path('login_post/', views.login_post),
    path('home_get/', views.home_get),
    path('view_profile/', views.view_profile),
    path('change_get/', views.change_get),
    path('change_post/', views.change_post),
    path('forgot_get/', views.forgot_get),
    path('forgot_post/', views.forgot_post),
    path('logout_page/', views.logout_page),
    path('upload_new_get/', views.upload_new_get),

    path('upload_new_post/', views.upload_new_post),
    path('upload_xg_get/', views.upload_xg_get),
    path('upload_xg_post/', views.upload_xg_post),

    path('upload_svm_get/', views.upload_svm_get),
    path('upload_svm_post/', views.upload_svm_post),



    path('random_visualization/', views.random_visualization),
    path('svm_visualization/',views.svm_visualization),
    path('xg_visualization/',views.xg_visualization),

]
