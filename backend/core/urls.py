# backend/apps/core/urls.py
from django.urls import path
from .views import VisionModelViewSet, InferenceJobViewSet, HelloWorldViewSet, MetricsAPIView

urlpatterns = [
    path("hello/", HelloWorldViewSet.as_view({"get": "list"}), name="hello-world"),

    # list  → /api/vision-models/
    path(
        "vision-models/",
        VisionModelViewSet.as_view({"get": "list"}),
        name="visionmodel-list",
    ),
    # detail → /api/vision-models/<pk>/
    path(
        "vision-models/<int:pk>/",
        VisionModelViewSet.as_view({"get": "retrieve"}),
        name="visionmodel-detail",
    ),

    path(
        "inference-jobs/",
        InferenceJobViewSet.as_view({"get": "list", "post": "create"}),
        name="inference-jobs",
    ),

    path(
        "metrics/",
        MetricsAPIView.as_view(),
        name="metrics"
    ),
]
