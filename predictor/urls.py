from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.home, name='home'),
    path('api/predict/', views.predict_api, name='predict_api'),
    path('submit-feedback/', views.submit_feedback, name='submit_feedback'),
    # Custom admin dashboard (staff only)
    path('dashboard/', views.dashboard, name='dashboard'),
    path('dashboard/login/', views.DashboardLoginView.as_view(), name='dashboard_login'),
    path('dashboard/logout/', views.dashboard_logout, name='dashboard_logout'),
]
