# Autoscio GPU worker

Runs the two self-hosted AI models behind Autoscio's AI Video Studio on **one RunPod GPU Pod**:

| Model | Job kinds | What it does | Licence |
|---|---|---|---|
| [Chatterbox Multilingual V3](https://github.com/resemble-ai/chatterbox) | `tts` | Speaks a script in a cloned voice from a 10–30 s sample. No training needed. 23 languages incl. English and Hindi. | MIT |
| [LTX-Video 0.9.8 distilled](https://github.com/Lightricks/LTX-Video) | `t2v`, `i2v` | Generates short video clips from a prompt, or animates a photo | Code Apache-2.0; check the weights licence before commercial launch |

```
Autoscio backend ──POST /jobs──▶ this worker (RunPod Pod, RTX 4090)
     ▲                                 │ 1. download inputs from S3 (presigned GET)
     │                                 │ 2. run the model on the GPU
     │                                 │ 3. upload result to S3 (presigned PUT)
     └────────── signed webhook ◀──────┘ 4. "done, s3Key = …"
     (+ GET /jobs/{id} as a fallback if a webhook is missed)
```

Media never passes through the API: inputs and outputs move through short-lived S3 presigned URLs, so no tenant data stays on the Pod.

---

## 1. Deploy on RunPod

### One-time setup

1. **GitHub**: push this repo. The **CI** workflow tests it, builds the image and pushes it to `ghcr.io/<owner>/autoscio-gpu-worker:latest` (~20–30 min the first time). If the repo is private, either make the package public (profile → Packages → autoscio-gpu-worker → Package settings → Change visibility) or add a RunPod registry credential (step 3).
2. **Generate the shared secret** once and keep it safe:
   ```bash
   openssl rand -hex 32
   ```
   It goes into the Pod as `WORKER_API_KEY` and into the backend as `GPU_WORKER_API_KEY`.
3. **RunPod**: create an account, add credit, create an API key (Settings → API Keys). *Private image only:* Settings → Container Registry Credentials → add `ghcr.io` with your GitHub username and a token with `read:packages`.

### Launch a Pod: option A, the button

1. Repo → Settings → Secrets and variables → Actions → add `RUNPOD_API_KEY` and `WORKER_API_KEY` (and optionally `HF_TOKEN`).
2. Actions → **Deploy Pod** → Run workflow (defaults: RTX 4090, Community cloud, 80 GB volume).
3. The run waits until both models are loaded and prints the Pod URL.

### Launch a Pod: option B, the console

1. RunPod → Pods → **Deploy** → **Community Cloud** → **RTX 4090**.
2. **Edit Template**:
   - Container image: `ghcr.io/<owner>/autoscio-gpu-worker:latest`
   - Container disk: 30 GB
   - Volume disk: **80 GB**, mounted at `/workspace`
   - Expose HTTP ports: **8000**
   - Environment variables: `WORKER_API_KEY=<the secret>`, `MODELS_DIR=/workspace/models`
3. Deploy. First boot downloads ~35 GB of weights (10–20 min). Watch the Pod's **Logs**.
4. Check `https://<podId>-8000.proxy.runpod.net/health`; wait for `"ready": true`.

### Connect the backend

```env
GPU_WORKER_URL=https://<podId>-8000.proxy.runpod.net
GPU_WORKER_API_KEY=<the same secret as WORKER_API_KEY>
```

### Day-to-day

| You want to… | Do this |
|---|---|
| Stop paying for the GPU | RunPod → Pod → **Stop**. Weights stay on the volume (~$0.20/GB/month while stopped). |
| Start again | **Start**. Ready in ~1–3 min, with no re-download. |
| Start fails ("no GPU available") | Deploy a **new** Pod (button or console). It re-downloads the weights (10–20 min). Update `GPU_WORKER_URL` in the backend, then **Terminate** the old Pod. |
| Ship new worker code | Push to `main` → CI builds `:latest` → deploy a new Pod (the image is pulled at Pod creation) → update `GPU_WORKER_URL` → terminate the old Pod. |
| Done for now | **Terminate** the Pod (deletes the volume too; next deploy re-downloads). |

On demo days, start the Pod an hour early and leave it running. Don't swap Pods minutes before a presentation.

---

## 2. API

Every endpoint except `/health` needs `Authorization: Bearer <WORKER_API_KEY>`.

### `POST /jobs`

Returns immediately (`202`, or `200` if that `taskId` was already submitted, so retries are safe).

```jsonc
{
  "taskId": "gpu_task_123",            // the backend's id; [A-Za-z0-9_-], ≤100 chars
  "kind": "tts",                       // "tts" | "t2v" | "i2v"
  "params": { ... },                   // see below
  "inputs": {
    "voiceSampleUrl": "https://…",     // tts: presigned GET of the tenant's voice sample (any audio format)
    "imageUrl": "https://…"            // i2v: presigned GET of the photo to animate
  },
  "output": {
    "putUrl": "https://…",             // presigned PUT, valid ≥ 2h (queue wait + render)
    "s3Key": "tenants/org1/…/a.mp3",
    "contentType": "audio/mpeg",       // tts: audio/mpeg | audio/wav ; t2v/i2v: video/mp4
    "headers": {}                      // any extra headers the PUT was signed with
  },
  "webhook": { "url": "https://api.autoscio.me/…" }   // optional
}
```

**`tts` params**

| Field | Default | Notes |
|---|---|---|
| `text` | — | ≤ 5000 chars. Long scripts are split on sentences and stitched. |
| `language` | `en` | `en, hi, es, fr, de, ar, ja, zh, …` (23 total) |
| `exaggeration` | `0.5` | 0–2. Higher = more expressive. |
| `cfgWeight` | `0.5` | 0–1. Lower = slower, more deliberate pacing. |

With no `voiceSampleUrl`, Chatterbox's built-in default voice is used.

**`t2v` / `i2v` params**

| Field | Default | Notes |
|---|---|---|
| `prompt` | — | ≤ 2000 chars. Detailed, literal scene descriptions work best. |
| `negativePrompt` | built-in | |
| `aspectRatio` | `9:16` | `9:16` (544×960), `16:9` (960×544), `1:1` (704×704) |
| `durationSeconds` | `5` | Clamped to `LTX_MAX_SECONDS` (default 5; what fits a 24 GB GPU) |
| `seed` | random | Fix it to reproduce a clip |

### `GET /jobs/{taskId}` and the webhook body

Both return the same shape:

```jsonc
{
  "taskId": "gpu_task_123",
  "kind": "tts",
  "status": "COMPLETED",        // QUEUED | RUNNING | COMPLETED | FAILED | CANCELLED
  "result": {                   // when COMPLETED
    "s3Key": "tenants/org1/…/a.mp3",
    "contentType": "audio/mpeg",
    "bytes": 482113,
    "durationSec": 31.4,
    "meta": { "chunks": 3, "language": "en", "clonedVoice": true }
  },
  "error": null,                // when FAILED: a human-readable reason
  "retryable": false,           // when FAILED: will resubmitting (fresh URLs) help?
  "createdAt": "…", "startedAt": "…", "finishedAt": "…",
  "timings": { "queuedMs": 120, "runMs": 41000 }   // runMs = GPU time, for metering
}
```

`404` means the worker doesn't know the job. Jobs live in memory, so a Pod restart forgets them. The backend should resubmit.

### Webhook signature

Each webhook carries `X-Worker-Timestamp` and `X-Worker-Signature: sha256=<hex>`, an HMAC-SHA256 of `"<timestamp>.<raw body>"` keyed with `WORKER_API_KEY`. Verify it against the **raw** body:

```ts
import { createHmac, timingSafeEqual } from "crypto";

function verify(rawBody: Buffer, timestamp: string, signature: string, key: string) {
  if (Math.abs(Date.now() / 1000 - Number(timestamp)) > 300) return false; // replay window
  const expected = "sha256=" + createHmac("sha256", key).update(`${timestamp}.`).update(rawBody).digest("hex");
  return expected.length === signature.length && timingSafeEqual(Buffer.from(expected), Buffer.from(signature));
}
```

Failed deliveries are retried 3 times (2 s, 5 s, 15 s), then given up. That's why the backend should also poll `GET /jobs/{id}` for tasks that have been quiet too long.

### `DELETE /jobs/{taskId}`

Cancels a `QUEUED` job. `409` if it is already running.

### `GET /health` (no auth)

```json
{ "ok": true, "ready": true, "runners": { "tts": { "state": "ready" }, "video": { "state": "ready" } }, "queueDepth": 0, "running": null, "version": "<git sha>" }
```

Runner states: `loading` (downloading or loading weights), `ready`, `error` / `crashed` (auto-restarts with backoff; see the Pod logs).

---

## 3. Configuration (Pod environment variables)

See [.env.example](.env.example). The ones that matter:

| Variable | Default | |
|---|---|---|
| `WORKER_API_KEY` | — | **Required.** The worker refuses to start without it. |
| `MODELS_DIR` | `/workspace/models` | Keep on the persistent volume. |
| `LTX_PIPELINE_CONFIG` | `ltxv-2b-0.9.8-distilled.yaml` | `ltxv-13b-0.9.8-distilled.yaml` = better quality, needs a 48 GB+ GPU (L40S / A6000). |
| `LTX_OFFLOAD_TO_CPU` | `1` | Parks the text encoder in CPU RAM during generation; needed on 24 GB GPUs. If videos still hit "GPU ran out of memory", lower `LTX_LONG_EDGE` (e.g. `768`). |
| `ENABLED_RUNNERS` | `tts,video` | Run just one model on a Pod. |

---

## 4. How it works inside the Pod

- **`app/main.py`**: FastAPI job API on `:8000` (the only exposed port).
- **`app/jobs.py`**: in-memory FIFO queue; one job at a time, since there's one GPU.
- **`app/runner_manager.py`**: starts and supervises one process per model and restarts it if it dies.
- **`app/runners/tts_runner.py`**, **`video_runner.py`**: load the model **once** and serve `127.0.0.1` only.

The two models run in **separate Python environments** (`/opt/venv-tts`, `/opt/venv-video`) because Chatterbox pins `transformers==5.2` and LTX-Video needs `transformers<4.52`. Both reuse the base image's PyTorch 2.6 / CUDA 12.4.

Weights are **not** in the image. They download on first boot into `$MODELS_DIR/hf` (Hugging Face cache, ~35 GB), shared by both runners.

## 5. Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements/api.txt -r requirements/dev.txt
ruff check . && ruff format --check . && pytest -q
```

Tests run with `RUNNER_MODE=fake`, so no GPU or model is needed. The real models only ever run on the Pod.
