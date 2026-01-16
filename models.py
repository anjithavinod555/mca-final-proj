from django.contrib.auth.models import User
from django.db import models

# Create your models here.

class Register(models.Model):
    name=models.CharField(max_length=100)
    mail=models.CharField(max_length=100)
    phone=models.CharField(max_length=100)
    USER=models.OneToOneField(User,on_delete=models.CASCADE)

class upload(models.Model):
    date=models.DateField()
    REGISTER=models.ForeignKey(Register,on_delete=models.CASCADE)
    input=models.CharField(max_length=300)
    result=models.CharField(max_length=300)