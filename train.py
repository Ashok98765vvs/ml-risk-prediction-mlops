"""Train on synthetic classification data, never real credit records."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

DIR=Path('artifacts');DIR.mkdir(exist_ok=True)
X,y=make_classification(n_samples=6000,n_features=6,n_informative=4,n_redundant=0,
                        weights=[.78,.22],flip_y=.02,random_state=42)
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
model=RandomForestClassifier(n_estimators=150,min_samples_leaf=4,random_state=42,n_jobs=-1)
model.fit(Xtrain,ytrain)
prob=model.predict_proba(Xtest)[:,1]
metrics={'accuracy':round(float(accuracy_score(ytest,prob>=.5)),4),
         'roc_auc':round(float(roc_auc_score(ytest,prob)),4),
         'test_rows':len(ytest),'synthetic_data':True}
joblib.dump(model,DIR/'model.joblib')
np.save(DIR/'training_reference.npy',Xtrain)
(DIR/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
