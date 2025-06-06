# core/admin.py
from django.contrib import admin
from .models import VisionModel, InferenceJob

@admin.register(VisionModel)
class VisionModelAdmin(admin.ModelAdmin):
    list_display  = ("name", "num_classes", "added_at")
    readonly_fields = ("added_at",)

@admin.register(InferenceJob)
class InferenceJobAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "vision_model", "status", "created_at", "updated_at")
    list_filter = ("status", "vision_model", "created_at")
    search_fields = ("id", "user__username", "vision_model__name")
    readonly_fields = ("id", "created_at", "updated_at", "error_message")