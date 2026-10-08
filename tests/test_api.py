import json
import time
import uuid

from app import transfer
from app.errors import JobFailure
from tests.conftest import AUTH


def tts_job(**overrides):
    job = {
        "taskId": f"task_{uuid.uuid4().hex[:12]}",
        "kind": "tts",
        "params": {"text": "Hello from Autoscio. Namaste!", "language": "en"},
        "inputs": {"voiceSampleUrl": "https://s3.test/voice.wav?sig=1"},
        "output": {
            "putUrl": "https://s3.test/out.mp3?sig=put",
            "s3Key": "tenants/org1/narration/a.mp3",
            "contentType": "audio/mpeg",
        },
        "webhook": {"url": "https://api.test/webhook/runpod"},
    }
    job.update(overrides)
    return job


def video_job(**overrides):
    job = tts_job(
        kind="t2v",
        params={"prompt": "A sunrise over a city skyline", "aspectRatio": "9:16"},
        inputs={},
        output={
            "putUrl": "https://s3.test/clip.mp4?sig=put",
            "s3Key": "tenants/org1/clips/a.mp4",
            "contentType": "video/mp4",
        },
    )
    job.update(overrides)
    return job


def wait_for(client, task_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = client.get(f"/jobs/{task_id}", headers=AUTH).json()
        if view["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return view
        time.sleep(0.02)
    raise AssertionError(f"job {task_id} did not finish: {view}")


def test_health_needs_no_auth_and_reports_runners(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True and body["ready"] is True
    assert set(body["runners"]) == {"tts", "video"}


def test_jobs_require_the_api_key(client):
    assert client.post("/jobs", json=tts_job()).status_code == 401
    bad = {"Authorization": "Bearer wrong"}
    assert client.post("/jobs", json=tts_job(), headers=bad).status_code == 401
    assert client.get("/jobs/whatever", headers=bad).status_code == 401


def test_tts_job_uploads_to_s3_and_sends_signed_webhook(client, internet):
    job = tts_job()
    res = client.post("/jobs", json=job, headers=AUTH)
    assert res.status_code == 202
    assert res.json()["status"] == "QUEUED"

    view = wait_for(client, job["taskId"])
    assert view["status"] == "COMPLETED", view
    assert view["result"]["s3Key"] == "tenants/org1/narration/a.mp3"
    assert view["result"]["contentType"] == "audio/mpeg"

    upload = internet.uploads["https://s3.test/out.mp3?sig=put"]
    assert upload["headers"]["content-type"] == "audio/mpeg"
    assert upload["body"] == b"fake-tts"

    # Webhook delivery happens just after the status flips; give it a moment.
    deadline = time.time() + 2
    while not internet.webhooks and time.time() < deadline:
        time.sleep(0.02)
    hook = internet.webhooks[0]
    payload = json.loads(hook.content)
    assert payload["taskId"] == job["taskId"] and payload["status"] == "COMPLETED"
    expected = transfer.sign("test-key", hook.headers["x-worker-timestamp"], hook.content)
    assert hook.headers["x-worker-signature"] == expected


def test_same_task_id_is_idempotent(client):
    job = video_job()
    first = client.post("/jobs", json=job, headers=AUTH)
    second = client.post("/jobs", json=job, headers=AUTH)
    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["taskId"] == job["taskId"]


def test_presigned_headers_are_forwarded_on_upload(client, internet):
    job = video_job()
    job["output"]["headers"] = {"x-amz-server-side-encryption": "AES256"}
    client.post("/jobs", json=job, headers=AUTH)
    assert wait_for(client, job["taskId"])["status"] == "COMPLETED"
    headers = internet.uploads["https://s3.test/clip.mp4?sig=put"]["headers"]
    assert headers["x-amz-server-side-encryption"] == "AES256"


def test_expired_upload_url_fails_as_retryable(client, internet):
    internet.put_status = 403
    job = video_job()
    client.post("/jobs", json=job, headers=AUTH)
    view = wait_for(client, job["taskId"])
    assert view["status"] == "FAILED"
    assert view["retryable"] is True
    assert "presigned PUT URL probably expired" in view["error"]


def test_unreachable_input_fails_cleanly(client):
    job = tts_job(inputs={"voiceSampleUrl": "https://s3.test/missing.wav?sig=1"})
    client.post("/jobs", json=job, headers=AUTH)
    view = wait_for(client, job["taskId"])
    assert view["status"] == "FAILED"
    assert "sig=" not in view["error"]  # presigned credentials never leak into errors


def test_runner_failure_is_reported_with_reason(client, monkeypatch):
    async def broken(kind, payload):
        raise JobFailure("Unsupported language 'xx'", retryable=False)

    monkeypatch.setattr(client.app.state.runners, "run", broken)
    job = tts_job()
    client.post("/jobs", json=job, headers=AUTH)
    view = wait_for(client, job["taskId"])
    assert view["status"] == "FAILED"
    assert view["error"] == "Unsupported language 'xx'"
    assert view["retryable"] is False


def test_validation_rejects_bad_requests(client):
    wrong_type = tts_job()
    wrong_type["output"]["contentType"] = "video/mp4"
    assert client.post("/jobs", json=wrong_type, headers=AUTH).status_code == 422

    empty_text = tts_job(params={"text": ""})
    assert client.post("/jobs", json=empty_text, headers=AUTH).status_code == 422

    i2v_without_image = video_job(kind="i2v")
    assert client.post("/jobs", json=i2v_without_image, headers=AUTH).status_code == 422

    bad_id = tts_job(taskId="../../etc/passwd")
    assert client.post("/jobs", json=bad_id, headers=AUTH).status_code == 422


def test_i2v_downloads_the_image(client, internet):
    job = video_job(kind="i2v", inputs={"imageUrl": "https://s3.test/photo?sig=1"})
    assert client.post("/jobs", json=job, headers=AUTH).status_code == 202
    assert wait_for(client, job["taskId"])["status"] == "COMPLETED"


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope", headers=AUTH).status_code == 404
    assert client.delete("/jobs/nope", headers=AUTH).status_code == 404
