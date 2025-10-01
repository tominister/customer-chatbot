# customer-chatbot

Small Flask-based chatbot with an intent classification model and optional LLM-backed responses.

This repository contains training scripts under `train/`, the Flask app (`app.py`), the `chatbot/` package, and a lightweight deployment pattern that keeps large model artifacts out of the Docker image.

## Quick start (local dev)

1. Create and activate a Python venv (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Flask app locally (development):

```powershell
python app.py
# visit http://127.0.0.1:5000
```

3. If you get import errors for the LLM client, set `GROQ_API_KEY` or leave it unset to use the fallback message in `chatbot/llm.py`.

## Recommended workflow for Docker-based deploys (no local large images)

This repo uses a pattern that keeps the Docker image small by excluding the `model/` directory and downloading the model at container startup. The `.dockerignore` file excludes `model/`, `data/`, and `performance/`.

Files added to support Docker deploys:
- `.dockerignore` – excludes local venv, model, data, and performance artifacts.
- `Dockerfile` – builds a small production image (does NOT include `model/`).
- `docker-compose.yml` – development compose that mounts a local model directory.
- `README-Docker.md` – detailed Docker and remote-build notes.

To run locally with Docker and a local model mounted:

```powershell
# build and run with docker-compose (will mount ./model into container)
docker compose up --build
```

Remote build & deploy (no local Docker storage):
- Render: connect GitHub, set start command to `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120` and set env vars in the dashboard.
- Google Cloud Run: `gcloud run deploy --source=. --region=us-central1 --allow-unauthenticated --set-env-vars FLASK_SECRET_KEY=...,MODEL_STORAGE_URL=...`

## Model hosting

- For large models, upload `intent_model.h5`, `tokenizer.pkl`, and `label_encoder.pkl` to cloud storage (S3/GCS) and set `MODEL_STORAGE_URL` to a signed URL or a private URL the container can access. The app will attempt to download the model at startup if not found locally.
- For local dev, place the model files in `./model/` (the repo includes a mount in `docker-compose.yml` for dev use).

## Environment variables

- `FLASK_SECRET_KEY` – secret for sessions; set in production.
- `GROQ_API_KEY` – optional LLM provider key. If omitted, the app will show a placeholder message for LLM responses.
- `MODEL_STORAGE_URL` – optional signed URL or storage URL to download the model at container startup.

## Health check

The app exposes a `/health` endpoint that returns 200 when the model and preprocessors are loaded and 503 while loading or missing. Configure your platform health checks to use this path.

## CI/CD & deployment notes

- Use remote builds when possible (Render, Cloud Run, Railway) to avoid building large images locally.
- Store secrets in the platform's secret manager rather than in `.env` files.

