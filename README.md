# Cracks Backend

Backend service for our Vision Transformer (ViT) platform developed in Django and Dockerized environment. 

## Features
- Structure damage inference using Vision Transformer models
- Django REST API backend
- User management
- Dockerized for easy development and deployment

## Setup
### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for local development)

### Quick Start (Docker)
```bash
docker-compose up --build
```
- Backend: http://localhost:8000/

### Local Development
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run Django backend:
   ```bash
   cd backend
   python manage.py migrate
   python manage.py runserver
   ```

## Usage
- API endpoints: see Django backend (`backend/core/urls.py`)
- Upload images and run inference via API or Django admin