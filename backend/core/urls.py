from django.urls import path
from .views import PhotoUploadView, GalleryView

urlpatterns = [
    path('upload/', PhotoUploadView.as_view(), name='upload_photo'),
    path('', GalleryView.as_view(), name='gallery'),
]
