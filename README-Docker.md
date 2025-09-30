# Docker deployment (small-image pattern)

This project supports a pattern where the Docker image does NOT contain large model files.
Instead, the container downloads the model at startup from a `MODEL_STORAGE_URL` (e.g., S3/GCS signed URL).

Why this pattern?
- Keeps the Docker image small and fast to build.
- Avoids committing large files to the repo or building huge images locally.

Files added:
- `Dockerfile` - builds a minimal image (does not copy `model/` due to `.dockerignore`).
- `.dockerignore` - excludes `model/`, `data/`, `performance/`, and `venv/`.
- `docker-compose.yml` - useful for local dev (mounts local model).
- `README-Docker.md` - this file.

Quick local dev (with Docker installed):
1. If you have a local model, mount it via compose and run:
   docker compose up --build
   # open http://localhost:8080

Remote build and deploy (no local Docker required):
- Use Render, Railway, or Google Cloud Run remote build features:
  - Render: connect GitHub repo, set start command to:
    gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
  - Cloud Run: use `gcloud run deploy --source=.` to let Cloud Build do the build remotely.

Model hosting:
- Upload your model `intent_model.h5`, `tokenizer.pkl`, and `label_encoder.pkl` to S3/GCS and set `MODEL_STORAGE_URL` (signed URL or private URL) in platform env vars.
- Alternatively, mount the model into the container for local dev using `docker-compose.yml`.

Environment variables to set in production:
- `FLASK_SECRET_KEY` - production secret
- `GROQ_API_KEY` - LLM provider key (if using LLM integration)
- `MODEL_STORAGE_URL` - signed URL to download the model at startup (optional)

Cleaning local Docker artifacts (if you build locally):
- docker system prune -a --volumes
- docker builder prune

If you'd like, I can also:
- Add a startup progress log or readiness probe to `app.py` while the model is downloading.
- Create a `render.yaml` for deploying to Render with git-based deploy.
