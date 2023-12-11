from fastapi.testclient import TestClient
from server_meta import app

client = TestClient(app)


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Upload Image" in response.text

def test_processing_request():
    files = {'file': ('test.jpg', open('data/zidane.jpg', 'rb'), 'image/jpeg')}
    data = {'model_name': 'yolov5s'}
    response = client.post("/", files=files, data=data)
    assert response.status_code == 200
    assert "object detected successfully" in response.json()["message"]

def test_processing_request_error():
    files = {'file': ('test.txt', open('data/bad_image.txt', 'rb'), 'text/plain')}
    data = {'model_name': 'yolov5s'}
    response = client.post("/", files=files, data=data)
    assert response.status_code == 400
    assert "object detection failed" in response.json()["message"]