"""Simple distribution-shift report comparing recent requests to synthetic training reference."""
from pathlib import Path
import json
import numpy as np
REF=Path('artifacts/training_reference.npy');LOG=Path('data/predictions.jsonl')
if not REF.exists() or not LOG.exists():
    raise SystemExit('Need a trained model and prediction logs first')
training=np.load(REF)
rows=[json.loads(line)['features'] for line in LOG.read_text().splitlines() if line.strip()]
if len(rows)<30:
    raise SystemExit('Collect >=30 prediction requests before making a drift report')
recent=np.asarray(rows[-500:]); report=[]
for i in range(training.shape[1]):
    base_std=float(np.std(training[:,i]))
    delta=abs(float(np.mean(recent[:,i])-np.mean(training[:,i]))) / max(base_std,1e-8)
    report.append({'feature':f'f{i}', 'mean_shift_in_training_std':round(delta,3),
                   'heuristic_alert':delta>0.5})
print(json.dumps({'recent_count':len(recent),'heuristic_only':True,'features':report},indent=2))
