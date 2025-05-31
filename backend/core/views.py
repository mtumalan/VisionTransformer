from django.views.generic.edit import CreateView
from django.views.generic.list import ListView
from django.urls import reverse_lazy
from .models import Photo
from .forms import PhotoForm


class PhotoUploadView(CreateView):
    model = Photo
    form_class = PhotoForm
    template_name = 'core/upload.html'
    success_url = reverse_lazy('gallery')


class GalleryView(ListView):
    model = Photo
    template_name = 'core/gallery.html'
    context_object_name = 'photos'
    ordering = ['-uploaded_at']
