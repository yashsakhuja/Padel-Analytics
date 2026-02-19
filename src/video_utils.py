"""
Video loading utilities.
Supports local files and YouTube URLs (downloaded via yt-dlp).
"""
import os
import cv2
import numpy as np
from typing import Tuple


def is_youtube_url(source: str) -> bool:
    """Check if the source string is a YouTube URL."""
    return any(domain in source for domain in ["youtube.com", "youtu.be"])


def download_youtube(
    url: str,
    output_dir: str = "raw",
    start_time: float = None,
    end_time: float = None,
) -> str:
    """
    Download a YouTube video to output_dir using yt-dlp.
    If start_time/end_time are provided, only download that section.
    Returns the absolute path to the downloaded MP4 file.
    """
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)

    # Include time range in filename so changing timestamps forces re-download
    if start_time is not None and end_time is not None:
        time_tag = f"_{int(start_time)}-{int(end_time)}s"
    else:
        time_tag = ""
    outtmpl = os.path.join(output_dir, f"%(title)s{time_tag}.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "quiet": False,
    }

    # Use --download-sections to clip long videos/livestreams
    if start_time is not None and end_time is not None:
        ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(start_time, end_time)]
        )
        ydl_opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # Ensure .mp4 extension
        base, ext = os.path.splitext(filename)
        if ext != ".mp4":
            filename = base + ".mp4"

    return os.path.abspath(filename)


def load_video(
    source: str,
    start_time: float = None,
    end_time: float = None,
) -> Tuple[cv2.VideoCapture, dict]:
    """
    Load a video from a local path or YouTube URL.
    If start/end times are given and source is YouTube, downloads only that section.
    Returns (cv2.VideoCapture, metadata_dict).
    """
    if is_youtube_url(source):
        print(f"Downloading YouTube video: {source}")
        path = download_youtube(source, start_time=start_time, end_time=end_time)
        print(f"Downloaded to: {path}")
    else:
        path = os.path.abspath(source)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0

    metadata = {
        "path": path,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_s": duration_s,
    }

    return cap, metadata


def read_first_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """
    Read and return the first frame of the video.
    Resets capture position back to frame 0 afterward.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame from video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frame
