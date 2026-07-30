"""
web.api.pipeline_steps — The clipping pipeline, store-agnostic.

This is the same sequence of steps that used to live directly in
``worker.py``, extracted so it can run inside a dedicated child process
(see ``job_runner.py``) instead of a thread in the main FastAPI process.
Progress, status, errors, and the final clip list are reported through
callback functions instead of calling into ``store`` directly, since the
job store lives in the parent process's memory.
"""

from __future__ import annotations

import glob
import json
import os
import time
import traceback
from typing import Callable, Optional

from .config_adapter import build_config_from_payload

TOTAL_STEPS = 7


def run(
    job_id: str,
    payload: dict,
    settings_env: dict,
    on_progress: Callable[..., None],
    on_status: Callable[[str], None],
    on_error: Callable[[str], None],
    on_clips: Callable[[list[dict]], None],
) -> None:
    try:
        cfg = build_config_from_payload(payload, job_id, env_overrides=settings_env)

        if not cfg.api_key_gemini:
            on_error("GOOGLE_API_KEY not found. Set it via Settings or the .env file.")
            return

        # --- Step 1: Download ---
        on_status("downloading")
        on_progress(
            step="download",
            step_number=1,
            total_steps=TOTAL_STEPS,
            message="Downloading video...",
            percent=5.0,
        )

        from clipping import engine

        source_platform = getattr(cfg, "source_platform", "youtube")

        if payload.get("upload_filename"):
            upload_path = os.path.join(os.getcwd(), "uploads", payload["upload_filename"])
            if not os.path.exists(upload_path):
                on_error(f"Uploaded file not found: {payload['upload_filename']}")
                return
            cfg.file_video_asli = upload_path
            on_progress(
                step="download",
                step_number=1,
                total_steps=TOTAL_STEPS,
                message="Using uploaded file.",
                percent=14.0,
            )
        else:
            if not cfg.url_youtube:
                if os.path.exists(cfg.file_video_asli):
                    on_progress(
                        step="download",
                        step_number=1,
                        total_steps=TOTAL_STEPS,
                        message="Skipping download: reusing existing video.",
                        percent=14.0,
                    )
                else:
                    on_error("Source video not found for that job ID. The file may have been deleted.")
                    return
            else:
                last_report = {"t": 0.0}

                def _on_download_progress(downloaded_bytes=0, total_bytes=None, speed=None, eta=None):
                    now = time.monotonic()
                    if now - last_report["t"] < 0.5:
                        return
                    last_report["t"] = now
                    if total_bytes:
                        pct = min(1.0, downloaded_bytes / total_bytes)
                        message = (
                            f"Downloading video... {downloaded_bytes / 1048576:.0f}/"
                            f"{total_bytes / 1048576:.0f} MB"
                        )
                    else:
                        pct = 0.0
                        message = f"Downloading video... {downloaded_bytes / 1048576:.0f} MB"
                    on_progress(
                        step="download",
                        step_number=1,
                        total_steps=TOTAL_STEPS,
                        message=message,
                        percent=5.0 + pct * 9.0,
                        download_bytes=downloaded_bytes,
                        download_total_bytes=total_bytes,
                        download_speed=speed,
                        download_eta=eta,
                    )

                video_metadata = engine.download_video(
                    cfg.url_youtube,
                    cfg.file_video_asli,
                    getattr(cfg, "use_dlp_subs", False),
                    getattr(cfg, "download_source_height", "max"),
                    source_platform=source_platform,
                    progress_callback=_on_download_progress,
                )
                on_progress(
                    step="download",
                    step_number=1,
                    total_steps=TOTAL_STEPS,
                    message="Video downloaded successfully.",
                    percent=14.0,
                    video_title=video_metadata.get("title"),
                    video_channel=video_metadata.get("channel"),
                    video_category=video_metadata.get("category"),
                )

        # --- Step 2: Transcribe ---
        on_status("transcribing")
        on_progress(
            step="transcribe",
            step_number=2,
            total_steps=TOTAL_STEPS,
            message="Starting transcription...",
            percent=15.0,
        )

        transkrip_lengkap = ""
        data_segmen = []

        json3_files = glob.glob(cfg.file_video_asli.replace(".mp4", ".*.json3"))
        file_json3 = json3_files[0] if json3_files else None

        if source_platform == "youtube" and getattr(cfg, "use_dlp_subs", False) and file_json3 and os.path.exists(file_json3):
            transkrip_lengkap, data_segmen = engine.parse_youtube_json3_subs(
                file_json3, max_words_per_subtitle=cfg.max_kata_per_subtitle
            )

        if not transkrip_lengkap or not data_segmen:
            def _on_transcribe_progress(transcribed_seconds=0.0, total_seconds=0.0):
                pct = min(1.0, transcribed_seconds / total_seconds) if total_seconds else 0.0
                on_progress(
                    step="transcribe",
                    step_number=2,
                    total_steps=TOTAL_STEPS,
                    message=(
                        f"Transcribing... {transcribed_seconds / 60:.1f}/"
                        f"{total_seconds / 60:.1f} min"
                    ),
                    percent=15.0 + pct * 20.0,
                    transcribe_seconds=transcribed_seconds,
                    transcribe_total_seconds=total_seconds,
                )

            transkrip_lengkap, data_segmen = engine.transcribe_video(
                cfg.file_video_asli,
                max_words_per_subtitle=cfg.max_kata_per_subtitle,
                model_size=cfg.whisper_model,
                device=cfg.whisper_device,
                compute_type=cfg.whisper_compute_type,
                progress_callback=_on_transcribe_progress,
            )

        # Caption fonts (Montserrat, Anton, Inter, Lora, Roboto, Bebas Neue)
        # have no Devanagari/Arabic/etc. glyph coverage -- burning in
        # native-script text renders every character as an empty tofu box.
        # Romanize just the caption word text (data_segmen); the AI
        # analysis step below still sees the original transkrip_lengkap,
        # since Gemini reads native script fine and it's the higher-
        # fidelity signal for picking highlight moments.
        data_segmen = engine.romanize_segments(data_segmen, cfg)

        on_progress(
            step="transcribe",
            step_number=2,
            total_steps=TOTAL_STEPS,
            message="Transcription complete.",
            percent=35.0,
        )

        # --- Step 3: AI Analysis ---
        on_status("analyzing")
        on_progress(
            step="analyze",
            step_number=3,
            total_steps=TOTAL_STEPS,
            message="Analyzing with AI...",
            percent=36.0,
        )

        gemini_output_path = os.path.join(cfg.outputs_dir, "gemini_response.json")

        if getattr(cfg, "mode", "auto") == "manual":
            # User specified the exact range themselves -- skip AI highlight
            # detection entirely. metadata.normalize_and_validate (step 4)
            # already tolerates missing titles/hashtags/scores (soft
            # defaults, nothing hard-required), so a single-item list with
            # just the timestamps is enough to flow through the rest of
            # the pipeline unchanged.
            hasil_json = [
                {
                    "rank": 1,
                    "start_time": cfg.manual_start_time,
                    "end_time": cfg.manual_end_time,
                    "title_indonesia": "",
                    "title_inggris": "",
                    "viral_score": 0,
                }
            ]
            on_progress(
                step="analyze",
                step_number=3,
                total_steps=TOTAL_STEPS,
                message="Using manually specified clip range.",
                percent=50.0,
            )
        else:
            if getattr(cfg, "load_gemini_json", False) and os.path.exists(gemini_output_path):
                with open(gemini_output_path, "r", encoding="utf-8") as f:
                    hasil_json = json.load(f)
            else:
                hasil_json = engine.analyze_with_ai(transkrip_lengkap, cfg)
                with open(gemini_output_path, "w", encoding="utf-8") as f:
                    json.dump(hasil_json, f, indent=4, ensure_ascii=False)

            on_progress(
                step="analyze",
                step_number=3,
                total_steps=TOTAL_STEPS,
                message=f"AI found {len(hasil_json)} candidate clips.",
                percent=50.0,
            )

        # --- Step 4: Metadata ---
        from clipping import metadata, studio

        hasil_json = metadata.normalize_and_validate(hasil_json)
        metadata_path = os.path.join(cfg.outputs_dir, "metadata_preview.json")
        metadata.save_metadata_preview(hasil_json, path=metadata_path)

        # Gemini selects candidates from the transcript alone -- it can't
        # see whether the footage itself is moving. Sample frames across
        # each candidate's range so a clip that reads well on audio but is
        # a paused/frozen screen on video doesn't get rendered ahead of a
        # genuinely dynamic one. Never dropped outright (a false positive
        # would silently lose a good clip) -- just sunk to the bottom.
        hasil_json = studio.filter_static_clips(cfg.file_video_asli, hasil_json)
        static_count = sum(1 for c in hasil_json if c.get("motion_check", {}).get("is_static"))
        hasil_json.sort(key=lambda c: c.get("motion_check", {}).get("is_static", False))
        # Re-number rank to match the new order -- the render loop later
        # sorts by rank, so the reorder above would otherwise be undone.
        for i, clip in enumerate(hasil_json):
            clip["rank"] = i + 1

        on_progress(
            step="metadata",
            step_number=4,
            total_steps=TOTAL_STEPS,
            message=(
                f"Metadata normalized ({static_count} static clip(s) flagged)."
                if static_count
                else "Metadata normalized."
            ),
            percent=55.0,
        )

        # --- Step 5: Diarization (optional) ---
        diarization_data = None
        from clipping import diarization as diarization_mod

        if (
            (getattr(cfg, "use_split_screen", False) and cfg.split_trigger == "diarization")
            or getattr(cfg, "use_camera_switch", False)
        ) and studio._is_vertical_ratio(cfg.pilihan_rasio):
            try:
                on_progress(
                    step="diarization",
                    step_number=5,
                    total_steps=TOTAL_STEPS,
                    message="Running speaker diarization...",
                    percent=56.0,
                )
                audio_path = cfg.file_video_asli.replace(".mp4", "_audio.wav")
                diarization_mod.extract_audio(cfg.file_video_asli, audio_path)
                num_speakers_arg = getattr(cfg, "diarization_num_speakers", 2)
                min_spk = None
                max_spk = None

                if str(num_speakers_arg).lower() == "auto":
                    max_faces = studio.estimate_speaker_count_from_video(cfg.file_video_asli, cfg)
                    num_speakers_arg = "auto"
                    min_spk = max(1, max_faces)
                    max_spk = min_spk + 2

                diarization_data = diarization_mod.run_diarization(
                    audio_path,
                    hf_token=cfg.hf_token,
                    num_speakers=num_speakers_arg,
                    min_speakers=min_spk,
                    max_speakers=max_spk,
                )
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                on_progress(
                    step="diarization",
                    step_number=5,
                    total_steps=TOTAL_STEPS,
                    message=f"Diarization failed: {e}. Falling back to standard mode.",
                    percent=58.0,
                )
                diarization_data = None

        # --- Step 6: Render Preparation ---
        on_status("rendering")
        on_progress(
            step="render",
            step_number=6,
            total_steps=TOTAL_STEPS,
            message="Preparing render...",
            percent=60.0,
        )

        os.environ["OSC_VIDEO_SCALE_ALGO"] = str(getattr(cfg, "video_scale_algo", "lanczos"))

        import cv2

        cap_e = cv2.VideoCapture(cfg.file_video_asli)
        src_h_e = int(cap_e.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_e.release()

        target_w_e, target_h_e = studio._get_render_dims(cfg, cfg.pilihan_rasio, source_h=src_h_e)
        video_encoder = studio.detect_video_encoder(cfg, target_h=target_h_e)

        file_glitch_ts = None
        if cfg.use_hook_glitch:
            cap_g = cv2.VideoCapture(cfg.file_video_asli)
            source_h_g = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_g.release()
            file_glitch_ts = studio.siapkan_glitch_video(
                cfg.pilihan_rasio, cfg, video_encoder, source_h=source_h_g
            )

        # --- Step 7: Render Each Clip ---
        from clipping import hook_manager

        render_manifest: list[dict] = []
        total_clips = len(hasil_json)

        custom_hook_path = None
        if getattr(cfg, "hook_source", None):
            custom_hook_path = hook_manager.download_custom_hook(cfg)

        for idx, klip in enumerate(sorted(hasil_json, key=lambda x: x["rank"])):
            clip_num = idx + 1
            on_progress(
                step="render",
                step_number=6,
                total_steps=TOTAL_STEPS,
                message=f"Rendering clip {clip_num}/{total_clips}...",
                percent=60.0 + (35.0 * clip_num / total_clips),
            )

            if custom_hook_path:
                klip["custom_hook_info"] = {"file_path": custom_hook_path}

            hasil_render = studio.proses_klip(
                klip["rank"],
                klip,
                cfg.pilihan_rasio,
                file_glitch_ts,
                data_segmen,
                cfg,
                video_encoder,
                diarization_data=diarization_data,
            )
            if hasil_render:
                render_manifest.append(hasil_render)

        # --- Save manifest ---
        manifest_path = os.path.join(cfg.outputs_dir, "render_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(render_manifest, f, ensure_ascii=False, indent=2)

        # --- Build clip details for the job store ---
        motion_by_rank = {c["rank"]: c.get("motion_check") for c in hasil_json}
        clips: list[dict] = []
        for entry in render_manifest:
            filename = os.path.basename(entry.get("output_file") or entry.get("video_path") or "")
            thumbnail_filename = os.path.basename(entry.get("thumbnail_path") or "")
            clips.append(
                {
                    "rank": entry.get("rank", 0),
                    "viral_score": entry.get("viral_score"),
                    "title": entry.get("title_indonesia", ""),
                    "title_en": entry.get("title_inggris", ""),
                    "filename": filename,
                    "duration": entry.get("duration"),
                    "start_time": entry.get("start_time"),
                    "end_time": entry.get("end_time"),
                    "download_url": f"/api/outputs/{job_id}/{filename}",
                    "thumbnail_url": (
                        f"/api/outputs/{job_id}/{thumbnail_filename}" if thumbnail_filename else None
                    ),
                    "motion_check": motion_by_rank.get(entry.get("rank", 0)),
                    "metadata": entry,
                }
            )

        on_clips(clips)
        on_progress(
            step="done",
            step_number=7,
            total_steps=TOTAL_STEPS,
            message=f"Done! {len(clips)} clips rendered successfully.",
            percent=100.0,
        )

    except Exception as exc:
        tb = traceback.format_exc()
        error_msg = f"{type(exc).__name__}: {exc}"
        on_error(error_msg)
        on_progress(
            step="error",
            step_number=0,
            total_steps=TOTAL_STEPS,
            message=f"Pipeline failed: {error_msg}",
            percent=0.0,
        )
        print(f"[job_runner] Job {job_id} failed:\n{tb}")
