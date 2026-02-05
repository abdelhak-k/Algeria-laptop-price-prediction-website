from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.home, name='home'),
    path('api/predict/', views.predict_api, name='predict_api'),
]
