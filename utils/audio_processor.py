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


def process_input(source: str) -> tuple:
    """
    Process input audio source into WAV chunks.
    Returns tuple: (chunks: list, main_wav_path: str)
    """
    is_valid, err_msg = validate_source_input(source)
    if not is_valid:
        logger.error(f"Input validation failed for '{source}': {err_msg}")
        raise ValueError(err_msg)

    if source.startswith("http://") or source.startswith("https://"):
        logger.info("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        logger.info("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    logger.info("Chunking audio...")
    chunks = chunk_audio(wav_path)
    logger.info(f"Audio processing ready — {len(chunks)} chunk(s) created.")
    return chunks, wav_path
