import os, uuid
from pathlib import Path
from django.db import models

class Photo(models.Model):
    title = models.CharField(max_length=100)
    image = models.ImageField(upload_to='photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

def weights_upload_to(_, filename):
    ext = Path(filename).suffix
    return f"weights/{uuid.uuid4().hex}{ext}"

class VisionModel(models.Model):
    name        = models.CharField(max_length=128, unique=True)
    description = models.TextField(blank=True)
    weights     = models.FileField(upload_to="weights/")
    num_classes = models.PositiveSmallIntegerField(default=2)
    input_size  = models.PositiveSmallIntegerField(default=224)
    added_at    = models.DateTimeField(auto_now_add=True)

    # *optional* –– if you want to keep the “parent” checkpoint for fine-tuning
    base_checkpoint = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="finetuned_versions"
    )

class TrainingJob(models.Model):
    """
    Optional queue entry if you want to train *inside* the platform.
    """
    created_at   = models.DateTimeField(auto_now_add=True)
    dataset_zip  = models.FileField(upload_to="datasets/")   # user-supplied zip (images/, masks/, csv)
    status       = models.CharField(max_length=24, default="queued",
                                    choices=[("queued","queued"),("running","running"),
                                             ("failed","failed"),("done","done")])
    log_path     = models.CharField(max_length=256, blank=True)
    result_model = models.ForeignKey(VisionModel, null=True, blank=True,
                                     on_delete=models.SET_NULL)

    def __str__(self):                     # e.g. “job-42 (running)”
        return f"job-{self.pk} ({self.status})"
