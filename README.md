# VisionTransformer Platform

An end-to-end semantic segmentation pipeline powered by Vision Transformer (ViT) models, featuring a Django REST API backend, Celery task queue, PostgreSQL & Redis, Dockerized deployment and Nginx reverse proxy.

---

## Overview

This repository demonstrates the backend solution for detecting structural damage in images using Vision Transformer models. Users can:

1. Submit images via a REST API.  
2. Enqueue inference jobs to Celery workers.  
3. Retrieve segmentation masks or classification results.  
4. Train and evaluate ViT-based models locally.

---

## Features

- **Vision Transformer (ViT)** for high-accuracy segmentation  
- **Django REST Framework** backend  
- **Celery + Redis** for asynchronous job processing  
- **PostgreSQL** for persistent storage  
- **Docker & Docker Compose** for reproducible environments  
- **Nginx** as reverse proxy and SSL terminator  
- User authentication & management  
- Ready-to-use training scripts (CE & PAED variants)  

---

## Tech Stack

- **Language & Frameworks:** Python 3.10, Django 5.2, Django REST Framework  
- **ML Libraries:** PyTorch, timm  
- **Task Queue:** Celery, Redis  
- **Database:** PostgreSQL 15  
- **Proxy:** Nginx 1.25-alpine  
- **Containerization:** Docker, Docker Compose 3.9  

---

## Prerequisites

- [Docker](https://www.docker.com/) >= 20.10  
- [Docker Compose](https://docs.docker.com/compose/) >= 1.29  
- Python 3.10 (for local model training)  
- `git`  

---

## Installation

### Environment Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/your-org/VisionTransformer.git
   cd VisionTransformer
   ```

2. Copy `.env.example` to `.env` and fill in values:
   ```bash
   cp .env.example .env
   ```
   ```dotenv
   POSTGRES_DB=postgres
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   DJANGO_SECRET_KEY=your_secret_key_here
   DJANGO_DEBUG=True
   ORCH_URL=http://localhost:8001/enqueue/
   ORCH_SHARED_TOKEN=your_shared_token
   ```

### Development

```bash
docker-compose -f docker-compose.dev.yml up --build
```

- **Backend** at `http://localhost:8000/`  

### Production

```bash
docker-compose up --build -d
```

- Exposes ports **80** and **443** via Nginx.  

---

## Configuration

All service configuration values are managed via environment variables defined in `.env`. Key variables:

- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` – PostgreSQL credentials  
- `DJANGO_SECRET_KEY` – Django secret key  
- `DJANGO_DEBUG` – `True` or `False`  
- `ORCH_URL` – URL for Celery enqueue endpoint  
- `ORCH_SHARED_TOKEN` – Shared token for API authentication  

---

## Usage

- **Run migrations** (inside backend container):
  ```bash
  docker-compose exec backend python manage.py migrate
  ```
- **Create superuser**:
  ```bash
  docker-compose exec backend python manage.py createsuperuser
  ```

---

## API Endpoints

See `localhost:8000/api/schema/swagger-ui/` for full list.

---

## Model Training

All training scripts assume Python 3.10 and the dependencies in `requirements.txt`.

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train CE model**:
   ```bash
   python model/CE/trainCurrentViTmodel.py
   ```

3. **Train PAED model**:
   ```bash
   python model/PAED/ViTscript.py
   ```

4. **Run tests**:
   ```bash
   python model/CE/testViTModel.py
   python model/PAED/ViTscriptTest.py
   ```
## Documentation

- **Documentation:** [`documentation.pdf`](documentation.pdf)  
- **Slides:** [`presentation.pdf`](presentation.pdf)  
