# backend/apps/core/views.py

from django.http import HttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from .models import VisionModel, InferenceJob
from .serializers import VisionModelSerializer, InferenceJobSerializer, HelloWorldSerializer, MetricSerializer
import requests
import threading
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from django.contrib.auth.models import User

@ensure_csrf_cookie
def get_csrf_token(request):
    """
    A GET to this URL forces Django to set a 'csrftoken' cookie.
    We don’t need to return JSON or HTML—just an HTTP 200 with the decorator.
    """
    return HttpResponse("CSRF cookie set")


SERVER_MODELOS_URL = "http://172.20.100.2:8000/api/process/"


class VisionModelViewSet(viewsets.ReadOnlyModelViewSet):
    """
    GET /api/vision-models/       → list all models
    GET /api/vision-models/{id}/  → retrieve one model
    """
    queryset = VisionModel.objects.all().order_by("name")
    serializer_class = VisionModelSerializer
    permission_classes = [permissions.AllowAny]


class InferenceJobViewSet(viewsets.ModelViewSet):
    """
    - list    (GET    /api/inference-jobs/)       → all jobs for the logged-in user
    - create  (POST   /api/inference-jobs/)       → new job (upload image + pick model)
    - retrieve(GET    /api/inference-jobs/{id}/)  → user’s own job detail
    - update  (PATCH  /api/inference-jobs/{id}/)  → (optional) orchestrator updates
    - destroy (DELETE /api/inference-jobs/{id}/)  → (optional)
    """

    serializer_class = InferenceJobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Each user sees only their own jobs
        qs = InferenceJob.objects.filter(user=self.request.user)

        # <── new: optional ?status=... filter
        status_param = self.request.query_params.get("status")
        if status_param:
            qs = qs.filter(status=status_param.upper())

        return qs

    def perform_create(self, serializer):
        """
        1) Save the new InferenceJob (serializer.save() sets vision_model via PK, and user via the serializer).
        2) Push the job off to the external server in a separate thread so the client does not wait.
        """
        job = serializer.save()  # This uses InferenceJobSerializer.create()

        # Grab the chosen model’s ID
        model_id = job.vision_model.id

        # Spawn a daemon thread to call the external model server
        threading.Thread(
            target=self.call_model_server,
            args=(job, model_id),
            daemon=True,
        ).start()

    def call_model_server(self, job, model_identifier):
        """
        POST to an external server:
          - job_id       (string)
          - model_id     (integer)
          - input_image  (file)
        """
        try:
            with job.input_image.open("rb") as img_file:
                files = {"input_image": img_file}
                data = {
                    "job_id": str(job.id),
                    "model_id": model_identifier,
                }
                response = requests.post(
                    SERVER_MODELOS_URL, files=files, data=data, timeout=60
                )
                if response.status_code != 200:
                    # Log any non-200 responses
                    print(
                        "Error calling model server:",
                        response.status_code,
                        response.text,
                    )
        except Exception as e:
            print("Error calling model server:", e)


class HelloWorldViewSet(viewsets.ViewSet):
    """
    A simple view to return a "Hello, World!" message.
    """
    def list(self, request):
        serializer = HelloWorldSerializer(data={"message": "Hello, World!"})
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
class MetricsAPIView(APIView):
    """
    GET /api/metrics/  → returns:
      {
        "total_photos_analyzed":   <count of all InferenceJob records>,
        "total_failures_detected": <count of InferenceJob where status == "DONE">,
        "total_users":             <count of all user accounts>
      }
    """
    permission_classes = [AllowAny]  # allow anyone to read these public metrics

    def get(self, request, *args, **kwargs):
        # 1) total photos analyzed = count of all InferenceJob entries
        total_photos = InferenceJob.objects.count()

        # 2) total failures detected = count of jobs with status="DONE"
        #    (or adjust if you prefer counting by some other field)
        total_failures = InferenceJob.objects.filter(status="DONE").count()

        # 3) total users = count of all User accounts
        total_users = User.objects.count()

        data = {
            "total_photos_analyzed": total_photos,
            "total_failures_detected": total_failures,
            "total_users": total_users,
        }
        serializer = MetricSerializer(data)
        return Response(serializer.data)