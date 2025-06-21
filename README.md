# Backend (FastAPI)

## Structure
- `api.py` — Main FastAPI app
- `utils.py` — Helper functions
- `requirements.txt` — Python dependencies
- `.env.example` — Example environment variables

## Running Locally
```sh
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Deploying to Render
- Push this folder to a GitHub repo
- Use the following start command on Render:
  ```
  uvicorn api:app --host 0.0.0.0 --port 10000
  ```
- Add your environment variables in the Render dashboard

---

**Do not commit your real `.env` file! Use `.env.example` for reference.**
