"""
clipping.studio.motion_check — Flags AI-selected clip candidates whose
video is visually static (e.g. a paused game screen, a frozen webcam
frame) even though the transcript around it reads as clip-worthy.

Gemini picks candidates from the transcript alone; it has no visibility
into whether the corresponding video frames actually move. A clip can be
funny/tense on audio while the footage itself is a paused menu or idle
screen -- this samples frames across the clip's time range and measures
frame-to-frame pixel difference to catch that case before spending a full
render on it.
"""

import cv2
import numpy as np

# Empirically: real motion (talking heads, gameplay, footage in general)
# produces mean frame differences well above this; a genuinely static
# frame (game paused, black screen, frozen webcam) sits well below it.
STATIC_FRAME_DIFF_THRESHOLD = 2.0
DEFAULT_SAMPLE_COUNT = 8


def check_clip_motion(
    video_path: str,
    start_time: float,
    end_time: float,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> dict:
    """
    Sample frames across [start_time, end_time] and measure how static
    the footage is.

    Returns
    -------
    dict
        ``{"is_static": bool, "mean_diff": float, "samples_read": int}``.
        ``is_static`` is False (assume motion) if the video can't be read
        at all, so a broken probe never blocks a render that would
        otherwise have succeeded.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"is_static": False, "mean_diff": None, "samples_read": 0}

    duration = max(end_time - start_time, 0.1)
    step = duration / sample_count

    prev_gray = None
    diffs = []

    for i in range(sample_count):
        t = start_time + i * step
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))  # downscale -- only need a coarse signal

        if prev_gray is not None:
            diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
            diffs.append(diff)

        prev_gray = gray

    cap.release()

    if not diffs:
        return {"is_static": False, "mean_diff": None, "samples_read": 0}

    mean_diff = sum(diffs) / len(diffs)
    return {
        "is_static": mean_diff < STATIC_FRAME_DIFF_THRESHOLD,
        "mean_diff": round(mean_diff, 3),
        "samples_read": len(diffs) + 1,
    }


def filter_static_clips(video_path: str, clips: list[dict]) -> list[dict]:
    """
    Run ``check_clip_motion`` over each clip candidate's time range and
    annotate it with the result. Static clips are kept in the list (never
    silently dropped -- a false positive shouldn't lose a good clip) but
    flagged via ``clip["motion_check"]`` so callers can re-rank, warn, or
    exclude them explicitly.
    """
    annotated = []
    for clip in clips:
        start = float(clip.get("start_time", 0.0))
        end = float(clip.get("end_time", start))
        result = check_clip_motion(video_path, start, end)
        clip = dict(clip)
        clip["motion_check"] = result
        annotated.append(clip)
    return annotated
