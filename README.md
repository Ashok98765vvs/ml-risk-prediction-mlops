# ML prediction API with basic MLOps monitoring (synthetic demo)

## Architecture
Synthetic training data -> sklearn train/test split -> RandomForest model -> offline metrics -> joblib artifact -> FastAPI `/predict` -> append-only JSONL request log -> `monitor.py` heuristic drift report.

**Important:** This is not a validated credit-risk model. Inputs are arbitrary synthetic numeric features. Do not use for lending, eligibility, or any high-stakes decisions. The monitoring script detects a simple input mean shift, not full data quality, model drift, or production observability.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
pytest -q
uvicorn api:app --reload --port 8000
```
Then:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"features":[0.5,-0.2,1.1,0.0,0.7,-1.2]}'
```
After 30+ calls, run `python monitor.py` to see basic mean-shift flags.

## Docker
```bash
docker build -t ml-demo .
docker run --rm -p 8000:8000 ml-demo
```
The Docker image trains the synthetic model at build time; a deployable ML pipeline should use versioned, immutable model artifacts from a model registry instead. For deployments with persistent logs, mount `data/` as a volume or use a managed telemetry service.

## Production readiness backlog
Model registry and lineage, dataset versioning, authentication, rate limiting, input schema and outlier guards, automated evaluation, CI/CD, metrics dashboards, retraining approval gates, privacy controls, and container vulnerability scans.
