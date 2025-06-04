# backend/apps/core/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import VisionModelViewSet, InferenceJobViewSet, HelloWorldViewSet

router = DefaultRouter()
router.register(r"vision-models", VisionModelViewSet, basename="visionmodel")
router.register(r"inference-jobs", InferenceJobViewSet, basename="inferencejob")
router.register(r"hello-world", HelloWorldViewSet, basename="helloworld")

urlpatterns = [
    path("api/v1/", include(router.urls)),
]