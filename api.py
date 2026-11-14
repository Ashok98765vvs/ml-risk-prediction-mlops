"""Prediction API for synthetic model; no real credit decisions."""
from pathlib import Path
from datetime import datetime,timezone
import json
import joblib
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

app=FastAPI(title='Synthetic ML Prediction API',version='0.1.0')
MODEL_PATH=Path('artifacts/model.joblib')
MODEL=joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
LOG=Path('data/predictions.jsonl')
class PredictionRequest(BaseModel):
    features: list[float]=Field(min_length=6,max_length=6)
@app.get('/health')
def health():
    return {'status':'ok' if MODEL is not None else 'model_missing','synthetic_model':True}
@app.post('/predict')
def predict(payload:PredictionRequest):
    if MODEL is None:
        raise HTTPException(503,'Run python train.py first')
    p=float(MODEL.predict_proba([payload.features])[0,1])
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a') as out:
        out.write(json.dumps({'timestamp':datetime.now(timezone.utc).isoformat(),
                              'features':payload.features,'probability':p})+'\n')
    return {'positive_class_probability':round(p,6),'predicted_class':int(p>=.5),
            'disclaimer':'Synthetic demo only; not valid for real lending decisions'}
