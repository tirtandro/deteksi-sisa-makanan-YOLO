"""
S3C-Smart-Canteen: API Tests
Usage: pytest tests/test_api.py -v
"""
import os, sys, json, pytest, numpy as np, cv2
from pathlib import Path
from io import BytesIO

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="module")
def app():
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    return flask_app

@pytest.fixture(scope="module")
def client(app):
    return app.test_client()

@pytest.fixture
def sample_image():
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)
    cv2.circle(img, (320, 320), 250, (220, 220, 220), -1)
    cv2.ellipse(img, (250, 280), (80, 50), 0, 0, 360, (200, 200, 255), -1)
    _, buffer = cv2.imencode(".jpg", img)
    return BytesIO(buffer.tobytes())

class TestHealthEndpoint:
    def test_health_check(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.get_json()["success"] is True

class TestClassesEndpoint:
    def test_get_classes(self, client):
        r = client.get("/api/classes")
        assert r.status_code == 200
        d = r.get_json()
        assert d["total_classes"] > 0

class TestAnalyzeEndpoint:
    def test_no_file(self, client):
        r = client.post("/api/analyze")
        assert r.status_code == 400

    def test_invalid_file_type(self, client):
        r = client.post("/api/analyze", data={"file": (BytesIO(b"x"), "t.txt")}, content_type="multipart/form-data")
        assert r.status_code == 400

    def test_valid_image(self, client, sample_image):
        r = client.post("/api/analyze", data={"file": (sample_image, "test.jpg")}, content_type="multipart/form-data")
        assert r.status_code == 200
        d = r.get_json()
        assert d["success"] is True
        assert "image_id" in d

class TestWeightEstimator:
    def test_estimate_weight(self):
        from core.weight_estimator import WeightEstimator
        est = WeightEstimator()
        results = est.estimate(
            detections=[{"class_id": 0, "class_name": "nasi", "confidence": 0.9,
                         "bbox": [100,100,300,300], "mask_area_pixels": 10000, "mask_binary": None}],
            scale_factor=0.04, plate_area_px=200000, is_custom_model=True)
        assert len(results) == 1
        assert results[0]["estimated_weight_grams"] > 0

class TestErrorHandlers:
    def test_404(self, client):
        assert client.get("/api/nonexistent").status_code == 404
    def test_405(self, client):
        assert client.get("/api/analyze").status_code == 405

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
