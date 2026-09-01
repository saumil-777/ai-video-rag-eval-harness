import os
import glob
import time
import logging
import threading
import yt_dlp
from pydub import AudioSegment
from core.config import DOWNLOADS_DIR, AUDIO_CHUNK_MINUTES, validate_source_input

logger = logging.getLogger(__name__)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Thread lock to prevent concurrent YouTube downloads from collisions / Streamlit reruns
_download_lock = threading.Lock()


def _cleanup_stale_part_files(dir_path: str = DOWNLOADS_DIR):
    """Safely cleans up stale .part or .ytdl temporary download files."""
    if not os.path.exists(dir_path):
        return
    for pattern in ["*.part", "*.ytdl", "*.webm.part", "*.m4a.part", "*.mp4.part"]:
        for file_path in glob.glob(os.path.join(dir_path, pattern)):
            try:
                os.remove(file_path)
                logger.info(f"Automatically cleaned stale download file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove stale file {file_path}: {e}")


def download_youtube_audio(url: str, max_retries: int = 3) -> str:
    """
    Downloads YouTube audio cleanly with Windows-compatible file locking protection.

    Protection mechanisms:
    1. Lock: `_download_lock` prevents concurrent Streamlit reruns/threads from downloading simultaneously.
    2. `nopart: True`: Prevents yt-dlp from creating `.part` files and calling buggy `os.rename(".part" -> final)` on Windows.
    3. `overwrites: True`: Overwrites stale files cleanly.
    4. Stale file cleanup: Pre-cleans any leftover `.part` / `.ytdl` files.
    5. Retry/Backoff: Retries up to `max_retries` with exponential backoff on transient Windows file lock errors.
    """
    with _download_lock:
        _cleanup_stale_part_files(DOWNLOADS_DIR)

        # Sanitized output template with video ID prefix to prevent collision
        output_template = os.path.join(DOWNLOADS_DIR, "%(id)s_%(title).50s.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "nopart": True,           # CRITICAL FOR WINDOWS: Avoid .part file rename WinError 32
            "overwrites": True,       # Safely overwrite existing files
            "updatetime": False,       # Do not attempt to modify file mtime after download
            "extractor_args": {
                "youtube": {
                    "player_client": ["mweb", "android", "ios", "web"],
                }
            },
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/128.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            "nocheckcertificate": True,
            "geo_bypass": True,
            "socket_timeout": 30,
            "retries": 3,
            "fragment_retries": 3,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    raw_filename = ydl.prepare_filename(info)
                    base, _ = os.path.splitext(raw_filename)
                    wav_path = base + ".wav"

                    # Verify expected wav file exists
                    if os.path.exists(wav_path):
                        _cleanup_stale_part_files(DOWNLOADS_DIR)
                        return wav_path

                    # Fallback: check most recent wav file created in DOWNLOADS_DIR
                    candidates = [os.path.join(DOWNLOADS_DIR, f) for f in os.listdir(DOWNLOADS_DIR) if f.endswith(".wav")]
                    if candidates:
                        wav_path = max(candidates, key=os.path.getmtime)
                        _cleanup_stale_part_files(DOWNLOADS_DIR)
                        return wav_path

                    raise FileNotFoundError(f"WAV audio extraction output not found for {url}")

            except Exception as ex:
                last_error = ex
                logger.warning(
                    f"Download attempt {attempt}/{max_retries} failed for '{url}': {ex}. "
                    f"Retrying..."
                )
                _cleanup_stale_part_files(DOWNLOADS_DIR)
                if attempt < max_retries:
                    time.sleep(1.0 * attempt)

        raise RuntimeError(f"Failed to download YouTube audio after {max_retries} attempt(s): {last_error}")


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16kHz
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = AUDIO_CHUNK_MINUTES) -> list:
    audio = AudioSegment.from_file(wav_path)
    if len(audio) == 0:
        raise ValueError("Audio file is empty (0 duration).")

    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        if len(chunk) < 1000 and len(chunks) > 0:
            continue

        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def cleanup_audio_files(file_paths: list = None, main_wav_path: str = None):
    """Safely delete temporary WAV chunks and downloaded/converted audio files."""
    targets = set()
    if file_paths:
        targets.update(file_paths)
    if main_wav_path:
        targets.add(main_wav_path)

    for path in targets:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logger.info(f"Cleaned up temporary file: {path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {path}: {e}")

    _cleanup_stale_part_files(DOWNLOADS_DIR)


import re

def extract_youtube_video_id(url: str) -> str:
    """Extract 11-character YouTube video ID from various URL formats."""
    if not url:
        return ""
    url = url.strip()
    if len(url) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url
    patterns = [
        r"(?:v=|\/)([a-zA-Z0-9_-]{11})(?:[&?\/].*)?$",
        r"youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"embed\/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def fetch_youtube_transcript(url: str) -> str:
    """
    Retrieves transcript/captions for a YouTube video via YouTube Transcript API
    and yt-dlp caption metadata endpoints (which operate over lightweight HTTP APIs
    permitted on cloud host IPs).
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract a valid YouTube video ID from '{url}'.")

    # Strategy 1: youtube_transcript_api
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        try:
            ytt = YouTubeTranscriptApi()
            snippets = ytt.fetch(video_id)
            texts = []
            for item in snippets:
                txt = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "")
                if txt:
                    texts.append(txt.replace("\n", " "))
            full_transcript = " ".join(texts).strip()
            if full_transcript:
                logger.info(f"Successfully retrieved YouTube transcript ({len(full_transcript)} chars) via YouTubeTranscriptApi.")
                return full_transcript
        except Exception as ex_api:
            logger.warning(f"YouTubeTranscriptApi fetch failed for video ID '{video_id}': {ex_api}. Trying subtitle fallback...")
    except ImportError:
        logger.warning("youtube_transcript_api not installed. Trying yt-dlp subtitle fallback...")

    # Strategy 2: yt-dlp subtitle / caption extraction metadata
    try:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "json3/vtt/srt",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get("subtitles") or info.get("automatic_captions") or {}
            lang_keys = [k for k in subs if k.startswith("en")] + list(subs.keys())
            if lang_keys:
                selected_lang = lang_keys[0]
                formats = subs[selected_lang]
                for fmt in formats:
                    sub_url = fmt.get("url")
                    if sub_url:
                        import requests
                        resp = requests.get(sub_url, timeout=15)
                        if resp.ok:
                            if fmt.get("ext") == "json3" or "json3" in sub_url:
                                data = resp.json()
                                events = data.get("events", [])
                                lines = []
                                for ev in events:
                                    segs = ev.get("segs", [])
                                    line = "".join([s.get("utf8", "") for s in segs]).strip()
                                    if line:
                                        lines.append(line)
                                text = " ".join(lines).strip()
                                if text:
                                    logger.info(f"Successfully retrieved YouTube transcript ({len(text)} chars) via yt-dlp captions.")
                                    return text
                            elif resp.text.strip():
                                clean_lines = [
                                    re.sub(r"<[^>]+>", "", line).strip()
                                    for line in resp.text.splitlines()
                                    if line.strip() and not line.startswith("WEBVTT") and "-->" not in line and not line.isdigit()
                                ]
                                text = " ".join(clean_lines).strip()
                                if text:
                                    logger.info(f"Successfully retrieved YouTube transcript ({len(text)} chars) via yt-dlp subtitle URL.")
                                    return text
    except Exception as ex_ytdlp:
        logger.warning(f"yt-dlp subtitle metadata extraction failed for '{url}': {ex_ytdlp}")

    raise RuntimeError(
        f"Unable to download audio or retrieve closed captions for YouTube video '{url}'. "
        "Direct media downloading is restricted on cloud host IPs (HTTP 403) and no public captions/transcripts "
        "were found for this video. Please upload an audio/video file directly or try a YouTube video with closed captions."
    )


def process_input(source: str) -> tuple:
    """
    Process input audio source into WAV chunks or direct transcript override.
    Returns tuple: (chunks: list, main_wav_path: str | None, transcript_override: str | None)
    """
    is_valid, err_msg = validate_source_input(source)
    if not is_valid:
        logger.error(f"Input validation failed for '{source}': {err_msg}")
        raise ValueError(err_msg)

    if source.startswith("http://") or source.startswith("https://"):
        logger.info("Detected YouTube URL. Attempting direct audio download...")
        try:
            wav_path = download_youtube_audio(source)
            logger.info("Chunking audio...")
            chunks = chunk_audio(wav_path)
            logger.info(f"Audio processing ready — {len(chunks)} chunk(s) created.")
            return chunks, wav_path, None
        except Exception as ex:
            logger.warning(
                f"Direct YouTube audio download failed for '{source}': {ex}. "
                f"Attempting cloud-compatible YouTube caption/transcript retrieval..."
            )
            try:
                transcript_text = fetch_youtube_transcript(source)
                return [], None, transcript_text
            except Exception as transcript_ex:
                logger.error(f"YouTube processing completely failed for '{source}': {transcript_ex}")
                raise RuntimeError(
                    f"Failed to process YouTube video: direct media downloading is restricted (HTTP 403) "
                    f"and caption retrieval failed ({transcript_ex}). "
                    f"Please upload an audio/video file directly or try a YouTube video with closed captions."
                ) from transcript_ex
    else:
        logger.info("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)
        logger.info("Chunking audio...")
        chunks = chunk_audio(wav_path)
        logger.info(f"Audio processing ready — {len(chunks)} chunk(s) created.")
        return chunks, wav_path, None
