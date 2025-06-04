# core/admin.py
from django.contrib import admin
from .models import VisionModel

@admin.register(VisionModel)
class VisionModelAdmin(admin.ModelAdmin):
    list_display  = ("name", "num_classes", "added_at")
    readonly_fields = ("added_at",)