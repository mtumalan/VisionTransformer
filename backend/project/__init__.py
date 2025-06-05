# backend/project/__init__.py

# Make sure Celery app is loaded whenever Django starts
from .celery import app as celery_app

__all__ = ("celery_app",)
