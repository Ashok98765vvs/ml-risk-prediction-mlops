from fastapi.testclient import TestClient
from api import app
client=TestClient(app)
def test_health():
    r=client.get('/health')
    assert r.status_code==200
    assert r.json()['synthetic_model'] is True
def test_validation():
    r=client.post('/predict',json={'features':[1,2]})
    assert r.status_code==422
