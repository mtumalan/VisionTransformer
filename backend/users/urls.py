# backend/users/urls.py
from django.urls import path
from .views import RegisterView, LoginView, CurrentUserView, LogoutView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('current-user/', CurrentUserView.as_view(), name='current-user'),
    path('logout/', LogoutView.as_view(), name='logout'),   
]
