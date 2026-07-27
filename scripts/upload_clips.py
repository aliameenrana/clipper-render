"""
scripts/upload_clips.py — Uploads rendered clips to Cloudflare R2 (S3-compatible
API) and reports the final, real download URLs back to the Worker.

Run as a separate workflow step after run_job.py completes, so a transient
upload failure doesn't get conflated with a rendering failure in the job's
error reporting.

Usage: python scripts/upload_clips.py <job_id>
"""

from __future__ import annotations

import json
import os
import sys

import boto3
import requests

WORKER_BASE_URL = os.environ["CLIPPER_WORKER_URL"].rstrip("/")
CALLBACK_SECRET = os.environ["CLIPPER_CALLBACK_SECRET"]
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ.get("R2_BUCKET", "clipper-outputs")
# Public R2 bucket URL (r2.dev subdomain or a custom domain) -- where
# uploaded clips are actually served from, distinct from the S3 API
# endpoint used to upload them.
R2_PUBLIC_BASE_URL = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")


def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def main() -> None:
    job_id = sys.argv[1]
    manifest_path = os.path.join(os.getcwd(), "outputs", job_id, "render_result.json")

    if not os.path.exists(manifest_path):
        print(f"[upload_clips] no render_result.json for job {job_id}, nothing to upload", flush=True)
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        clips = json.load(f)

    client = get_r2_client()
    outputs_dir = os.path.join(os.getcwd(), "outputs", job_id)

    for clip in clips:
        filename = clip["filename"]
        local_path = os.path.join(outputs_dir, filename)
        if not os.path.exists(local_path):
            print(f"[upload_clips] WARNING: {local_path} not found, skipping", flush=True)
            continue

        r2_key = f"{job_id}/{filename}"
        print(f"[upload_clips] uploading {filename} -> r2://{R2_BUCKET}/{r2_key}", flush=True)
        client.upload_file(local_path, R2_BUCKET, r2_key)

        clip["download_url"] = f"{R2_PUBLIC_BASE_URL}/{r2_key}"

    # Re-report the final clip list with real R2 URLs, replacing the
    # placeholder download_urls sent by run_job.py's on_clips callback.
    res = requests.post(
        f"{WORKER_BASE_URL}/api/render-jobs/{job_id}/clips",
        json={"clips": clips},
        headers={"Authorization": f"Bearer {CALLBACK_SECRET}", "Content-Type": "application/json"},
        timeout=10,
    )
    if not res.ok:
        print(f"[upload_clips] final clips callback failed: {res.status_code} {res.text}", flush=True)
        sys.exit(1)

    res = requests.post(
        f"{WORKER_BASE_URL}/api/render-jobs/{job_id}/status",
        json={"status": "completed"},
        headers={"Authorization": f"Bearer {CALLBACK_SECRET}", "Content-Type": "application/json"},
        timeout=10,
    )
    if not res.ok:
        print(f"[upload_clips] status callback failed: {res.status_code} {res.text}", flush=True)
        sys.exit(1)

    print(f"[upload_clips] done, {len(clips)} clip(s) uploaded", flush=True)


if __name__ == "__main__":
    main()
