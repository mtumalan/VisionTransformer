# core/admin.py
from django.contrib import admin
from .models import VisionModel, TrainingJob

@admin.register(VisionModel)
class VisionModelAdmin(admin.ModelAdmin):
    list_display  = ("name", "num_classes", "added_at")
    readonly_fields = ("added_at",)

@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ("id", "status", "created_at", "result_model")
    readonly_fields = ("status", "log_path", "result_model", "created_at")