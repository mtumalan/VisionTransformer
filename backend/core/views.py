from django.views.generic.edit import CreateView
from django.views.generic.list import ListView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.urls import reverse_lazy
from .models import Photo, VisionModel
from .forms import PhotoForm

import requests
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


class AnalyzeView(APIView):
    def post(self, request):
        image = request.FILES.get('image')
        model_name = request.data.get('model_name')

        if not image or not model_name:
            return Response({'error': 'Missing required fields.'}, status=status.HTTP_400_BAD_REQUEST)

        photo = Photo.objects.create(title=f"_upload", image=image)

        try:
            vision_model = VisionModel.objects.get(name=model_name)
        except VisionModel.DoesNotExist:
            return Response({'error': 'Model not found.'}, status=status.HTTP_404_NOT_FOUND)

        external_url = "http://external-instance/analyze"  # TODO: use actual external service URL
        files = {'image': photo.image.open('rb')}
        data = {
            'model_name': vision_model.name,
        }

        try:
            response = requests.post(external_url, files=files, data=data)
            response.raise_for_status()
            external_result = response.json()
        except Exception as e:
            return Response({'error': f'External request failed: {str(e)}'}, status=status.HTTP_502_BAD_GATEWAY)

        return Response(external_result, status=status.HTTP_200_OK)


class VisionModelListView(APIView):
    def get(self):
        models = VisionModel.objects.all()
        data = [
            {
                'id': m.id,
                'name': m.name,
                'description': m.description,
                'num_classes': m.num_classes,
                'input_size': m.input_size,
                'added_at': m.added_at,
            }
            for m in models
        ]
        return Response(data)
