import os
import tempfile

import httpx
import pytest
from fastapi.testclient import TestClient

os.environ.update(
    {
        "RUNNER_MODE": "fake",
        "WORKER_API_KEY": "test-key",
        "WORK_DIR": tempfile.mkdtemp(prefix="gpu-worker-test-"),
    }
)

from app import transfer  # noqa: E402
from app.main import app  # noqa: E402

AUTH = {"Authorization": "Bearer test-key"}


class FakeInternet:
    """Stands in for S3 presigned URLs and the backend webhook."""

    def __init__(self):
        self.uploads: dict[str, dict] = {}
        self.webhooks: list[httpx.Request] = []
        self.put_status = 200
        self.downloads = {
            "https://s3.test/voice.wav?sig=1": (b"RIFFfakewav", "audio/wav"),
            "https://s3.test/photo?sig=1": (b"\x89PNGfake", "image/png"),
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "GET" and url in self.downloads:
            body, ctype = self.downloads[url]
            return httpx.Response(200, content=body, headers={"content-type": ctype})
        if request.method == "GET":
            return httpx.Response(403)
        if request.method == "PUT":
            self.uploads[url] = {"body": request.content, "headers": dict(request.headers)}
            return httpx.Response(self.put_status)
        if request.method == "POST" and url.startswith("https://api.test/webhook"):
            self.webhooks.append(request)
            return httpx.Response(200)
        return httpx.Response(404)


@pytest.fixture
def internet():
    return FakeInternet()


@pytest.fixture
def client(internet, monkeypatch):
    monkeypatch.setattr(transfer, "transport_override", httpx.MockTransport(internet.handler))
    monkeypatch.setattr(transfer, "WEBHOOK_BACKOFF_S", (0,))
    with TestClient(app) as test_client:
        yield test_client
