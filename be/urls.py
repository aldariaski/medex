from django.contrib import admin
from django.urls import include, path
from django.shortcuts import redirect

from . import views
app_name = 'be'

urlpatterns = [
    path('', views.home, name='home'),
    path('results/', views.results, name='results'),
]
