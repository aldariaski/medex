from django.contrib import admin
from django.urls import include, path
from django.shortcuts import redirect

from . import views
app_name = 'fe'

urlpatterns = [
    path('', views.home, name='home'),
    path('results/', views.results, name='results'),
    path('faq/', views.faq, name='faq'),
    path('indeksi/', views.indeksi, name='indeksi'),
]
