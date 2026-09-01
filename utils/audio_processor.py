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
    and yt-dlp caption metadata endpoints, prioritizing English (including YouTube's
    native auto-translated English tracks) whenever available.
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract a valid YouTube video ID from '{url}'.")

    # Strategy 1: youtube_transcript_api (English direct / translation)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt = YouTubeTranscriptApi()

        # 1a. Try English directly
        try:
            snippets = ytt.fetch(video_id, languages=["en", "en-US", "en-GB"])
            texts = [
                getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "")
                for item in snippets
            ]
            full_transcript = " ".join([t.replace("\n", " ") for t in texts if t]).strip()
            if full_transcript:
                logger.info(f"Successfully retrieved English YouTube transcript ({len(full_transcript)} chars) via YouTubeTranscriptApi.")
                return full_transcript
        except Exception as ex_en:
            logger.warning(f"YouTubeTranscriptApi English fetch failed for video ID '{video_id}': {ex_en}. Trying transcript list & translation...")

        # 1b. Transcript list search (for native translation or direct English tracks)
        try:
            t_list = ytt.list(video_id)
            for t in t_list:
                if getattr(t, "is_translatable", False):
                    try:
                        translated = t.translate("en").fetch()
                        texts = [
                            getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "")
                            for item in translated
                        ]
                        full_transcript = " ".join([txt.replace("\n", " ") for txt in texts if txt]).strip()
                        if full_transcript:
                            logger.info(f"Successfully retrieved native English-translated YouTube transcript ({len(full_transcript)} chars) via YouTubeTranscriptApi.")
                            return full_transcript
                    except Exception:
                        pass
        except Exception as ex_list:
            logger.warning(f"YouTubeTranscriptApi list search failed for video ID '{video_id}': {ex_list}. Trying yt-dlp subtitle fallback...")
    except ImportError:
        logger.warning("youtube_transcript_api not installed. Trying yt-dlp subtitle metadata fallback...")

    # Strategy 2: yt-dlp subtitle / caption extraction metadata (includes YouTube's auto-translated tracks like 'en')
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
            subs = info.get("subtitles") or {}
            auto_subs = info.get("automatic_captions") or {}

            # Prioritize 'en' tracks (manual or auto-translated)
            en_keys = [k for k in subs if k.startswith("en")] + [k for k in auto_subs if k.startswith("en")]
            all_keys = en_keys + [k for k in subs if k not in en_keys] + [k for k in auto_subs if k not in en_keys]

            for key in all_keys:
                formats = subs.get(key) or auto_subs.get(key) or []
                for fmt in formats:
                    sub_url = fmt.get("url")
                    if not sub_url:
                        continue
                    import requests
                    resp = requests.get(sub_url, timeout=15)
                    if resp.ok and resp.text.strip():
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
                                logger.info(f"Successfully retrieved YouTube transcript ({len(text)} chars, lang='{key}') via yt-dlp captions.")
                                return text
                        elif "<text" in resp.text:
                            raw_lines = re.findall(r"<text[^>]*>(.*?)</text>", resp.text, re.DOTALL)
                            clean_lines = [
                                re.sub(r"<[^>]+>", "", l).replace("&quot;", '"').replace("&amp;", "&").replace("&#39;", "'").strip()
                                for l in raw_lines
                            ]
                            text = " ".join([l for l in clean_lines if l]).strip()
                            if text:
                                logger.info(f"Successfully retrieved YouTube transcript ({len(text)} chars, lang='{key}') via yt-dlp XML captions.")
                                return text
                        else:
                            clean_lines = [
                                re.sub(r"<[^>]+>", "", line).strip()
                                for line in resp.text.splitlines()
                                if line.strip() and not line.startswith("WEBVTT") and "-->" not in line and not line.isdigit()
                            ]
                            text = " ".join(clean_lines).strip()
                            if text:
                                logger.info(f"Successfully retrieved YouTube transcript ({len(text)} chars, lang='{key}') via yt-dlp subtitle URL.")
                                return text
    except Exception as ex_ytdlp:
        logger.warning(f"yt-dlp subtitle metadata extraction failed for '{url}': {ex_ytdlp}")

    # Strategy 3: Fallback to raw transcript in any language via youtube_transcript_api if no English track exists anywhere
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt = YouTubeTranscriptApi()
        snippets = ytt.fetch(video_id)
        texts = [
            getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "")
            for item in snippets
        ]
        full_transcript = " ".join([t.replace("\n", " ") for t in texts if t]).strip()
        if full_transcript:
            logger.info(f"Retrieved raw fallback transcript ({len(full_transcript)} chars) via YouTubeTranscriptApi.")
            return full_transcript
    except Exception:
        pass

    raise RuntimeError(
        f"Unable to access this YouTube video. Please try another public YouTube video."
    )


def process_input(source: str) -> tuple:
    """
    Process input audio source into WAV chunks or direct transcript override.
    For YouTube URLs, tries closed captions FIRST before falling back to audio download.
    Returns tuple: (chunks: list, main_wav_path: str | None, transcript_override: str | None)
    """
    is_valid, err_msg = validate_source_input(source)
    if not is_valid:
        logger.error(f"Input validation failed for '{source}': {err_msg}")
        raise ValueError(err_msg)

    if source.startswith("http://") or source.startswith("https://"):
        logger.info("Detected YouTube URL. Attempting closed-caption / transcript retrieval first...")
        # PREFERRED PATH: Try closed captions / transcript retrieval first
        try:
            transcript_text = fetch_youtube_transcript(source)
            if transcript_text and transcript_text.strip():
                logger.info(f"Successfully retrieved YouTube transcript ({len(transcript_text)} chars). Skipping audio download.")
                return [], None, transcript_text
        except Exception as caption_ex:
            logger.warning(
                f"YouTube closed-caption retrieval failed for '{source}': {caption_ex}. "
                f"Attempting direct audio download fallback..."
            )

        # FALLBACK PATH: Direct media audio download (if no public captions exist)
        try:
            wav_path = download_youtube_audio(source)
            logger.info("Chunking audio...")
            chunks = chunk_audio(wav_path)
            logger.info(f"Audio processing ready — {len(chunks)} chunk(s) created.")
            return chunks, wav_path, None
        except Exception as download_ex:
            logger.error(f"YouTube processing completely failed for '{source}': {download_ex}")
            raise RuntimeError(
                "Unable to access this YouTube video. Please try another public YouTube video."
            ) from download_ex
    else:
        logger.info("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)
        logger.info("Chunking audio...")
        chunks = chunk_audio(wav_path)
        logger.info(f"Audio processing ready — {len(chunks)} chunk(s) created.")
        return chunks, wav_path, None
