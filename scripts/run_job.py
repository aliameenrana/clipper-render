"""
scripts/run_job.py — GitHub Actions entrypoint for the clip-rendering pipeline.

Invoked by .github/workflows/render.yml via `workflow_dispatch` inputs.
Runs the same store-agnostic pipeline.run() used locally, but reports
progress/results to the deployed clipper-suggestions Worker over the real
network (not loopback HTTP, since this runs on a GitHub-hosted runner),
authenticated with a shared secret. Rendered clips are uploaded to R2 by
scripts/upload_clips.py as a separate step, after this script finishes.

Usage: python scripts/run_job.py <job_id> <payload_json>
"""

from __future__ import annotations

import json
import os
import sys
import time

# Running as `python scripts/run_job.py` only puts scripts/ on sys.path,
# not the repo root -- add it explicitly so `from pipeline import
# pipeline_steps` (a sibling of scripts/, not of this file) resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

WORKER_BASE_URL = os.environ["CLIPPER_WORKER_URL"].rstrip("/")
CALLBACK_SECRET = os.environ["CLIPPER_CALLBACK_SECRET"]


def _callback_headers() -> dict:
    return {"Authorization": f"Bearer {CALLBACK_SECRET}", "Content-Type": "application/json"}


def _post_with_retry(path: str, body: dict, attempts: int = 3) -> None:
    url = f"{WORKER_BASE_URL}{path}"
    last_exc = None
    for attempt in range(attempts):
        try:
            res = requests.post(url, json=body, headers=_callback_headers(), timeout=10)
            if res.ok:
                return
            print(f"[run_job] callback {path} returned {res.status_code}: {res.text}", flush=True)
        except requests.RequestException as exc:
            last_exc = exc
            print(f"[run_job] callback {path} attempt {attempt + 1} failed: {exc}", flush=True)
        time.sleep(2 * (attempt + 1))
    if last_exc:
        print(f"[run_job] callback {path} gave up after {attempts} attempts: {last_exc}", flush=True)


def on_progress(job_id: str, **event) -> None:
    _post_with_retry(f"/api/render-jobs/{job_id}/progress", event)


def on_status(job_id: str, status: str) -> None:
    _post_with_retry(f"/api/render-jobs/{job_id}/status", {"status": status})


def on_error(job_id: str, error: str) -> None:
    _post_with_retry(f"/api/render-jobs/{job_id}/error", {"error": error})


def on_clips(job_id: str, clips: list[dict]) -> None:
    # Clip download_url values are placeholders at this point -- the
    # separate upload_clips.py step rewrites them to real R2 URLs and
    # re-reports the final clip list after uploading.
    _post_with_retry(f"/api/render-jobs/{job_id}/clips", {"clips": clips})
    # Also write to a local file so upload_clips.py (the next workflow step)
    # knows exactly what was rendered and where, without re-deriving it.
    manifest_path = os.path.join(os.getcwd(), "outputs", job_id, "render_result.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=2)


def main() -> None:
    job_id = sys.argv[1]
    payload = json.loads(sys.argv[2])

    # Any failure here -- including an import error before pipeline_steps
    # even runs -- must still reach the Worker, or the job is stuck showing
    # "queued"/"downloading" forever with no error surfaced anywhere.
    try:
        settings_env = {
            "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "PEXELS_API_KEY": os.environ.get("PEXELS_API_KEY", ""),
        }

        from pipeline import pipeline_steps

        pipeline_steps.run(
            job_id=job_id,
            payload=payload,
            settings_env=settings_env,
            on_progress=lambda **event: on_progress(job_id, **event),
            on_status=lambda status: on_status(job_id, status),
            on_error=lambda error: on_error(job_id, error),
            on_clips=lambda clips: on_clips(job_id, clips),
        )
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        print(f"[run_job] fatal error before/outside pipeline_steps.run(): {tb}", flush=True)
        on_error(job_id, f"{type(exc).__name__}: {exc}")
        on_status(job_id, "failed")
        raise


if __name__ == "__main__":
    main()
