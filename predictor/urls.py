from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.home, name='home'),
    path('suggest/', views.suggest, name='suggest'),
    path('compare/', views.compare, name='compare'),
    path('api/predict/', views.predict_api, name='predict_api'),
    path('submit-feedback/', views.submit_feedback, name='submit_feedback'),
    # Custom admin dashboard (staff only)
    path('dashboard/', views.dashboard, name='dashboard'),
    path('dashboard/feedback/', views.dashboard_feedback, name='dashboard_feedback'),
    path('dashboard/login/', views.DashboardLoginView.as_view(), name='dashboard_login'),
    path('dashboard/logout/', views.dashboard_logout, name='dashboard_logout'),
    # Error page previews (visible in any DEBUG mode)
    path('errors/400/', views.error_400, name='error_400_preview'),
    path('errors/403/', views.error_403, name='error_403_preview'),
    path('errors/404/', views.error_404, name='error_404_preview'),
    path('errors/500/', views.error_500, name='error_500_preview'),
]
