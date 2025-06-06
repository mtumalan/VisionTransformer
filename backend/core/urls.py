# backend/apps/core/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import VisionModelViewSet, InferenceJobViewSet, HelloWorldViewSet

urlpatterns = [
    path('hello/', HelloWorldViewSet.as_view({'get': 'list'}), name='hello-world'),
    path('vision-models/', VisionModelViewSet.as_view({'get': 'list', 'get': 'retrieve'}), name='vision-models'),
    path('inference-jobs/', InferenceJobViewSet.as_view({'get': 'list', 'post': 'create'}), name='inference-jobs'),
]