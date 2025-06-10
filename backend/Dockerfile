# backend/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ─── 1) Install system dependencies (including postgresql-client) ───
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      postgresql-client \
 && rm -rf /var/lib/apt/lists/*

# ─── 2) Copy & install Python dependencies ───
COPY docker/weights/epoch=99-step=81092.ckpt /opt/weights/epoch=99-step=81092.ckpt
COPY docker/weights/calss_names_colors.csv  /opt/weights/calss_names_colors.csv

COPY requirements.txt /app/
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ─── 3) Copy project files (including wait-for-db.sh) ───
COPY . /app/

# Ensure wait-for-db.sh is executable
RUN chmod +x /app/wait-for-db.sh

# ─── 4) ENTRYPOINT to wait for DB then run whatever command is passed ───
# We do not hardcode user/password here; script will pick them up from the env.
ENTRYPOINT ["/app/wait-for-db.sh"]

# ─── 5) Default CMD for development (override in docker-compose if needed) ───
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]
