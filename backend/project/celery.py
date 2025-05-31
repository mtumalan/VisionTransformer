# backend/project/celery.py
import os
from celery import Celery

# Ensure the Django settings module is set
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project.settings")

app = Celery("project")
# Read celery settings, using namespace="CELERY" to pull from settings.py
app.config_from_object("django.conf:settings", namespace="CELERY")
# Auto-discover tasks in each installed app’s "tasks.py"
app.autodiscover_tasks()
