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


class InferenceJobSerializer(serializers.ModelSerializer):
    """
    For listing/creating a job. 
    - On create, the user submits: { vision_model: <id>, input_image: <file> }.
    - Read-only fields: status, mask_image, created_at, updated_at, error_message.
    """

    # We may want to return a nested VisionModel representation
    vision_model_details = VisionModelSerializer(source="vision_model", read_only=True)
    
    class Meta:
        model = InferenceJob
        read_only_fields = [
            "id", "user", "status", "mask_image", "created_at", "updated_at", "error_message",
            "vision_model_details",
        ]
        fields = [
            "id",
            "user",
            "vision_model",
            "vision_model_details",
            "input_image",
            "mask_image",
            "status",
            "error_message",
            "created_at",
            "updated_at",
        ]

    def create(self, validated_data):
        """
        Override create() to set `user` from request context.
        """
        request = self.context.get("request")
        user = request.user
        # vision_model and input_image are already in validated_data
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