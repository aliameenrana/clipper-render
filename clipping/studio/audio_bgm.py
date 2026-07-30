"""
audio_bgm.py — Local BGM asset resolver.

Reads MP3 files from assets/bgm/<mood>/ directories and selects one at random.
"""

import os
import random


def get_local_bgm_file(mood, bgm_dir):
    """
    Get a random BGM MP3 file from the local assets directory based on mood.

    Args:
        mood (str): The requested mood (e.g., 'chill', 'epic', 'sad').
        bgm_dir (str): Base directory for BGM assets.

    Returns:
        str: Absolute path to the selected MP3 file, or None if not found/empty.
    """
    mood_dir = os.path.join(bgm_dir, mood)

    if not os.path.exists(mood_dir) or not os.path.isdir(mood_dir):
        return None

    mp3_files = [f for f in os.listdir(mood_dir) if f.lower().endswith(".mp3")]

    if not mp3_files:
        return None

    selected_file = random.choice(mp3_files)
    return os.path.abspath(os.path.join(mood_dir, selected_file))


def build_bgm_filter(
    bgm_mode, bgm_base_volume, audio_input_voc="[1:a]", audio_input_bgm="[2:a]",
    fade_in_sec=0.0, fade_out_sec=0.0, clip_duration=None,
):
    """
    Build the FFmpeg filter_complex string for BGM mixing.

    Args:
        bgm_mode (str): 'ducking' for sidechain compress, 'background' for constant volume mix.
        bgm_base_volume (float): Base volume level for BGM (e.g. 0.25).
        audio_input_voc (str): FFmpeg stream label for vocal audio input.
        audio_input_bgm (str): FFmpeg stream label for BGM audio input.
        fade_in_sec (float): BGM fade-in duration in seconds, 0 to disable.
        fade_out_sec (float): BGM fade-out duration in seconds, 0 to disable.
        clip_duration (float): Total clip duration in seconds; required for fade_out_sec > 0
            since afade's fade-out start point is measured from the beginning of the stream.

    Returns:
        str: The filter_complex string for FFmpeg.
    """
    voc_format = f"{audio_input_voc}aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume=1.2[voc]"

    bgm_fade_parts = []
    if fade_in_sec > 0:
        bgm_fade_parts.append(f"afade=t=in:st=0:d={fade_in_sec}")
    if fade_out_sec > 0 and clip_duration:
        fade_out_start = max(0.0, clip_duration - fade_out_sec)
        bgm_fade_parts.append(f"afade=t=out:st={fade_out_start}:d={fade_out_sec}")
    bgm_fade = ("," + ",".join(bgm_fade_parts)) if bgm_fade_parts else ""

    bgm_format = (
        f"{audio_input_bgm}aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"volume={bgm_base_volume}{bgm_fade}[bgm]"
    )

    if bgm_mode == "background":
        # Simple constant-volume mix — no sidechain, BGM stays at bgm_base_volume throughout
        return (
            f"{voc_format}; "
            f"{bgm_format}; "
            f"[voc][bgm]amix=inputs=2:duration=first[a_out]"
        )
    else:
        # Ducking mode (default) — sidechain compress makes BGM duck under vocals
        return (
            f"{voc_format}; "
            f"{bgm_format}; "
            f"[voc]asplit=2[voc_sc][voc_mix]; "
            f"[bgm][voc_sc]sidechaincompress=threshold=0.08:ratio=5.0:attack=100:release=1000[bgm_ducked]; "
            f"[voc_mix][bgm_ducked]amix=inputs=2:duration=first[a_out]"
        )
