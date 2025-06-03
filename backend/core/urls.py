from django.urls import path
from .views import PhotoUploadView, GalleryView, AnalyzeView, VisionModelListView

urlpatterns = [
    path('upload/', PhotoUploadView.as_view(), name='upload_photo'),
    path('', GalleryView.as_view(), name='gallery'),
	path('analyze/', AnalyzeView.as_view(), name='analyze_photo'),
    path('models/', VisionModelListView.as_view(), name='list_models'),
]
