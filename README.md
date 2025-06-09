# Cracks Backend

Backend service for our Vision Transformer (ViT) platform developed in Django and Dockerized environment. 

## Features
- Structure damage inference using Vision Transformer models
- Django REST API backend
- User management
- Dockerized for easy development and deployment

## Environment Variables
To run the backend, create a `.env` file in the project root with the following content:

```env
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
DJANGO_SECRET_KEY=your‐django‐secret‐key
DJANGO_DEBUG=True
```

## Setup
### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for local development)

### Quick Start (Docker)
```bash
 docker-compose -f .\docker-compose.dev.yml up -d --build
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
