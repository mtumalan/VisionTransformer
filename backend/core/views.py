# backend/apps/core/views.py

from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import VisionModel, InferenceJob
from .serializers import VisionModelSerializer, InferenceJobSerializer, HelloWorldSerializer

class VisionModelViewSet(viewsets.ReadOnlyModelViewSet):
    """
    GET /api/v1/vision-models/          → list all models
    GET /api/v1/vision-models/{id}/     → retrieve one model
    """
    queryset = VisionModel.objects.all().order_by("name")
    serializer_class = VisionModelSerializer
    permission_classes = [permissions.AllowAny]


class InferenceJobViewSet(viewsets.ModelViewSet):
    """
    - list (GET     /api/v1/inference-jobs/)       → all jobs for the logged‐in user
    - create (POST  /api/v1/inference-jobs/)       → new job (upload image + pick model)
    - retrieve (GET /api/v1/inference-jobs/{id}/)  → user’s own job detail (see status, mask_image URL)
    - update/partial_update (PATCH) /api/v1/inference-jobs/{id}/ → (optional) you might allow the orchestrator to PATCH
    - destroy (DELETE) … maybe never needed for public
    """

    serializer_class = InferenceJobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Each user only sees their own jobs
        return InferenceJob.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # The serializer.create() method already pulls in `user=request.user`
        serializer.save()

    @action(detail=True, methods=["post"], permission_classes=[permissions.IsAdminUser])
    def complete(self, request, pk=None):
        """
        (Optional) If you prefer giving the orchestrator
        a single “/complete/” endpoint to upload the mask image:
        
        POST /api/v1/inference-jobs/{job_uuid}/complete/
        { mask_image: <file> }
        
        This action sets status="DONE" and saves mask_image.
        You could also accept a JSON URL, etc.
        """

        job = self.get_object()
        if job.status not in ["PENDING", "PROCESSING"]:
            return Response(
                {"error": "Job is already completed or failed."},
                status=status.HTTP_400_BAD_REQUEST
            )

        mask_file = request.FILES.get("mask_image")
        if not mask_file:
            return Response(
                {"error": "mask_image file is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        job.mask_image = mask_file
        job.status = "DONE"
        job.save(update_fields=["mask_image", "status", "updated_at"])
        return Response(
            InferenceJobSerializer(job, context={"request": request}).data,
            status=status.HTTP_200_OK
        )

class HelloWorldViewSet(viewsets.ViewSet):
    """
    A simple view to return a "Hello, World!" message.
    This is just an example to show how to create a custom viewset.
    """
    
    def list(self, request):
        serializer = HelloWorldSerializer(data={"message": "Hello, World!"})
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data, status=status.HTTP_200_OK)