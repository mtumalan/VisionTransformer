# backend/apps/core/utils.py   (or wherever process_job_local lives)

from django.core.files.base import ContentFile
from django.utils import timezone
import logging, traceback

from .inference import run_segmentation_outputs
from .models     import InferenceJob

logger = logging.getLogger(__name__)

def process_job_local(job_id: int, image_bytes: bytes) -> None:
    job = InferenceJob.objects.get(pk=job_id)

    # optional RUNNING state so the UI can show progress
    job.status = "RUNNING"
    job.updated_at = timezone.now()
    job.save(update_fields=["status", "updated_at"])

    try:
        outs      = run_segmentation_outputs(image_bytes=image_bytes)   # ← NEW
        mask_png  = outs["mask.png"]

        job.mask_image.save(f"mask_{job_id}.png",
                            ContentFile(mask_png), save=False)
        job.status      = "DONE"
        job.updated_at  = timezone.now()
        job.save(update_fields=["mask_image", "status", "updated_at"])

    except Exception as exc:
        logger.error("process_job_local failed for job %s – %s\n%s",
                     job_id, exc, traceback.format_exc())

        job.status        = "ERROR"
        job.error_message = str(exc)[:500]         # so the front end can display it
        job.updated_at    = timezone.now()
        job.save(update_fields=["status", "error_message", "updated_at"])
