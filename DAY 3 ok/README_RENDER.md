Deploying to Render (simple)
=================================

This project is a small Flask app that serves static files and a couple of API endpoints.

Prerequisites
-------------
- A Render account (https://render.com) connected to your GitHub/GitLab repository.
- Your repo pushed to GitHub (or GitLab).
- `GCP_API_KEY` (Google Cloud API key) for the Translate API — do NOT commit this to git.

Files added
-----------
- `requirements.txt` — lists runtime dependencies including `gunicorn`.
- `Procfile` — tells Render how to start the web process.

Steps to deploy
---------------
1. Commit and push this repository to GitHub.
2. In Render, create a new Web Service and connect your GitHub repo.
3. For Build Command, leave empty (Render will use pip install -r requirements.txt).
4. For Start Command, either leave empty (Render will detect the `Procfile`) or set:

   gunicorn app:app --bind 0.0.0.0:$PORT

5. In the Render service settings, add an environment variable:

   Key: GCP_API_KEY
   Value: <YOUR_GCP_API_KEY>

6. Deploy. Render will build and start the service. The app will be accessible at the Render-provided URL.

Testing locally (quick)
-----------------------
You can test locally with the `py` launcher or your venv:

PowerShell (using the working interpreter):

```
py -3 app.py
```

Or create & use a venv, then run with gunicorn:

```
py -3 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:5000
```

Security note
-------------
Do not store `GCP_API_KEY` or other secrets in the repository. Use Render environment variables.
