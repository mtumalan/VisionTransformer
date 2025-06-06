# backend/apps/core/models.py

import uuid
from django.conf import settings
from django.db import models

User = settings.AUTH_USER_MODEL

class Photo(models.Model):
    """
    (Optional) If you just want to store generic images, you can keep this.
    In this design, however, we'll embed the uploaded image directly on the InferenceJob.
    You may still use Photo if you need a separate “album” of images per user.
    """
    owner       = models.ForeignKey(User, on_delete=models.CASCADE, related_name="photos")
    title       = models.CharField(max_length=100)
    image       = models.ImageField(upload_to='photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} ({self.owner.username} @ {self.uploaded_at.isoformat()})"


class VisionModel(models.Model):
    """
    Pre‐registered AI models that users can choose from.
    For example, you might load these at admin time or via a management command.
    """
    name        = models.CharField(max_length=128, unique=True)
    description = models.TextField(blank=True)
    num_classes = models.PositiveSmallIntegerField(default=2)
    input_size  = models.PositiveSmallIntegerField(default=224)
    added_at    = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class InferenceJob(models.Model):
    """
    Each time a user uploads an image and picks a VisionModel,
    we create one of these. The “mask_image” is filled in later
    by the orchestrator once the AI finishes.
    """
    STATUS_CHOICES = [
        ("PENDING",   "Pending"),
        ("PROCESSING","Processing"),
        ("DONE",      "Done"),
        ("FAILED",    "Failed"),
    ]

    id             = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user           = models.ForeignKey(User, on_delete=models.CASCADE, related_name="inference_jobs")
    vision_model   = models.ForeignKey(VisionModel, on_delete=models.PROTECT, related_name="jobs")
    input_image    = models.ImageField(upload_to="inference_inputs/")
    mask_image     = models.ImageField(upload_to="inference_masks/", blank=True, null=True)
    status         = models.CharField(max_length=12, choices=STATUS_CHOICES, default="PENDING")
    created_at     = models.DateTimeField(auto_now_add=True)
    updated_at     = models.DateTimeField(auto_now=True)
    error_message  = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Job {self.id} by {self.user.username} [{self.status}]"