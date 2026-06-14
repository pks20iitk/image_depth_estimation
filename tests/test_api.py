"""
Test suite for the Depth Estimation API.

Run:
    pytest tests/ -v

For CI (no GPU, no real models):
    pytest tests/ -v -m "not integration"
"""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
def make_dummy_image(width: int = 640, height: int = 480) -> bytes:
    """Create a random RGB PIL image and return its JPEG bytes."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_mock_registry(depth_type: str = "relative"):
    """
    Build a mock DepthModelRegistry that returns a synthetic DepthResult
    without loading any real model weights.
    """
    from app.models.depth_model import DepthResult

    dummy_depth = np.random.rand(480, 640).astype(np.float32)

    mock_result = DepthResult(
        depth_map=dummy_depth,
        depth_type=depth_type,
        output_unit="normalized [0-1]" if depth_type == "relative" else "metres",
        model_name="mock_model",
        checkpoint="mock/checkpoint",
        inference_time_ms=42.0,
        original_size=(480, 640),
    )

    backend = MagicMock()
    backend.predict.return_value = mock_result
    backend.is_loaded = True
    backend.name = "mock_model"
    backend.checkpoint = "mock/checkpoint"
    backend.depth_type = depth_type
    backend.output_unit = mock_result.output_unit

    registry = MagicMock()
    registry.get.return_value = backend
    registry.list_models.return_value = []
    return registry


@pytest.fixture
def client_relative():
    """TestClient with mocked registry (relative depth)."""
    from app.main import app

    with patch.object(app, "state", MagicMock(registry=make_mock_registry("relative"))):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_metric():
    """TestClient with mocked registry (metric depth)."""
    from app.main import app

    with patch.object(app, "state", MagicMock(registry=make_mock_registry("metric"))):
        with TestClient(app) as c:
            yield c


# ─────────────────────────────────────────────
# Image utility tests  (unit)
# ─────────────────────────────────────────────
class TestImageUtils:
    def test_decode_valid_jpeg(self):
        from app.utils.image import decode_upload
        raw = make_dummy_image()
        img = decode_upload(raw, content_type="image/jpeg")
        assert img.mode == "RGB"
        assert img.width == 640

    def test_decode_rejects_too_large(self):
        from app.utils.image import decode_upload
        with pytest.raises(ValueError, match="exceeds limit"):
            decode_upload(b"x" * (21 * 1024 * 1024), max_size_bytes=20 * 1024 * 1024)

    def test_decode_rejects_bad_mime(self):
        from app.utils.image import decode_upload
        raw = make_dummy_image()
        with pytest.raises(ValueError, match="Unsupported content type"):
            decode_upload(raw, content_type="application/pdf")

    def test_resize_max_edge_downscales(self):
        from app.utils.image import resize_to_max_edge
        img = Image.new("RGB", (2000, 1000))
        resized = resize_to_max_edge(img, max_edge=800)
        assert max(resized.width, resized.height) == 800

    def test_resize_max_edge_noop_when_small(self):
        from app.utils.image import resize_to_max_edge
        img = Image.new("RGB", (400, 300))
        result = resize_to_max_edge(img, max_edge=800)
        assert result.size == (400, 300)

    def test_encode_depth_to_png_returns_bytes(self):
        from app.utils.image import encode_depth_to_png
        depth = np.random.rand(100, 100).astype(np.float32)
        png = encode_depth_to_png(depth, colormap="inferno")
        assert isinstance(png, bytes)
        assert png[:4] == b"\x89PNG"   # PNG magic bytes

    def test_depth_statistics_shape(self):
        from app.utils.image import depth_statistics
        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        stats = depth_statistics(depth)
        assert stats["min_depth"] == pytest.approx(1.0)
        assert stats["max_depth"] == pytest.approx(4.0)
        assert stats["mean_depth"] == pytest.approx(2.5)


# ─────────────────────────────────────────────
# Config tests  (unit)
# ─────────────────────────────────────────────
class TestConfig:
    def test_default_settings(self):
        from app.config import Settings
        s = Settings()
        assert s.max_upload_mb == 20
        assert s.colormap == "inferno"
        assert s.device in ("cpu", "cuda")

    def test_max_upload_bytes(self):
        from app.config import Settings
        s = Settings(max_upload_mb=5)
        assert s.max_upload_bytes == 5 * 1024 * 1024


# ─────────────────────────────────────────────
# Schemas tests  (unit)
# ─────────────────────────────────────────────
class TestSchemas:
    def test_depth_response_serialises(self):
        from app.schemas import DepthResponse, DepthStats
        resp = DepthResponse(
            model="test",
            depth_type="relative",
            output_unit="normalized [0-1]",
            original_size=[480, 640],
            depth_size=[480, 640],
            stats=DepthStats(min_depth=0.1, max_depth=0.9, mean_depth=0.5, std_depth=0.2),
            inference_time_ms=50.0,
        )
        d = resp.model_dump()
        assert d["depth_type"] == "relative"
        assert d["stats"]["min_depth"] == pytest.approx(0.1)


# ─────────────────────────────────────────────
# API endpoint tests  (integration — needs mocked registry)
# ─────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_ok(self):
        from app.main import app
        mock_registry = MagicMock()
        mock_registry.list_models.return_value = []
        app.state.registry = mock_registry
        with TestClient(app) as c:
            resp = c.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestDepthEndpoints:
    def _upload(self, client, endpoint: str, format_: str = "png"):
        raw = make_dummy_image()
        return client.post(
            f"/api/v1/depth/{endpoint}",
            params={"response_format": format_},
            files={"file": ("test.jpg", raw, "image/jpeg")},
        )

    def test_relative_png_response(self, client_relative):
        resp = self._upload(client_relative, "relative", "png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_relative_json_response(self, client_relative):
        resp = self._upload(client_relative, "relative", "json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["depth_type"] == "relative"
        assert "depth_base64" in body
        assert body["inference_time_ms"] > 0

    def test_metric_png_response(self, client_metric):
        resp = self._upload(client_metric, "metric", "png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_metric_json_response(self, client_metric):
        resp = self._upload(client_metric, "metric", "json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["depth_type"] == "metric"
        assert body["output_unit"] == "metres"

    def test_missing_file_returns_422(self, client_relative):
        resp = client_relative.post("/api/v1/depth/relative")
        assert resp.status_code == 422

    def test_invalid_format_returns_422(self, client_relative):
        raw = make_dummy_image()
        resp = client_relative.post(
            "/api/v1/depth/relative",
            params={"response_format": "xml"},
            files={"file": ("test.jpg", raw, "image/jpeg")},
        )
        assert resp.status_code == 422
