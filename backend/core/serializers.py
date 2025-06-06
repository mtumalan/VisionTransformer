# backend/apps/core/serializers.py

from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import VisionModel, InferenceJob

User = get_user_model()

class VisionModelSerializer(serializers.ModelSerializer):
    """
    List all available VisionModels. No create/update from the public API.
    """
    class Meta:
        model = VisionModel
        fields = ["id", "name", "description", "num_classes", "input_size", "added_at"]

class MetricSerializer(serializers.Serializer):
    total_photos_analyzed   = serializers.IntegerField()
    total_failures_detected = serializers.IntegerField()
    total_users             = serializers.IntegerField()
    
class InferenceJobSerializer(serializers.ModelSerializer):
    """
    For listing/creating a job:
      - Client POSTs: { vision_model: <id>, input_image: <file> }
      - Read‐only on GET: status, mask_image, created_at, updated_at, error_message, vision_model_details, user_username.
    """

    # 1) Nested details of the chosen VisionModel (read only)
    vision_model_details = VisionModelSerializer(source="vision_model", read_only=True)

    # 2) Expose the uploader's username (read only)
    user_username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = InferenceJob

        # Fields that should not be written by client (read only)
        read_only_fields = [
            "id",
            "user",
            "status",
            "mask_image",
            "created_at",
            "updated_at",
            "error_message",
            "vision_model_details",
            "user_username",
        ]

        # Fields that the API returns / consumes
        fields = [
            "id",
            "user",              # numeric FK (still available if needed)
            "user_username",     # <— newly added read‐only field
            "vision_model",
            "vision_model_details",
            "input_image",
            "mask_image",
            "status",
            "error_message",
            "created_at",
            "updated_at",
        ]

        read_only_fields = ("user",)

    def create(self, validated_data):
        """
        Override create() to set `user` from the request, then create the InferenceJob.
        """
        request = self.context.get("request")
        user = request.user
        job = InferenceJob.objects.create(user=user, **validated_data)
        return job

class HelloWorldSerializer(serializers.Serializer):
    """
    Simple serializer for a hello world message.
    """
    message = serializers.CharField(default="Hello, World!")
    
    def validate_message(self, value):
        if not value:
            raise serializers.ValidationError("Message cannot be empty.")
        return value