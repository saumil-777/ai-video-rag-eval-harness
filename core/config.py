import os
import re
from pathlib import Path

# ── LLM & Model Defaults ──────────────────────────────────────────────────────
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# ── Text Processing & Chunking Configuration ──────────────────────────────────
SUMMARIZATION_CHUNK_SIZE = 3000
SUMMARIZATION_CHUNK_OVERLAP = 200

RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
RETRIEVER_K = 4

# ── Audio Processing Configuration ───────────────────────────────────────────
AUDIO_CHUNK_MINUTES = 10
SARVAM_PIECE_SECONDS = 25
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"

# ── Directories ───────────────────────────────────────────────────────────────
DOWNLOADS_DIR = "downloads"
CHROMA_DIR = "vector_db"
DEFAULT_COLLECTION_NAME = "meeting_transcript"

# Supported audio/video file extensions
SUPPORTED_FILE_EXTENSIONS = {".mp4", ".wav", ".mp3", ".m4a", ".mkv", ".flac", ".ogg", ".webm", ".aac"}


def validate_source_input(source: str) -> tuple[bool, str]:
    """
    Validate input YouTube URL or local file path.
    Returns (is_valid: bool, error_message: str)
    """
    if not source or not source.strip():
        return False, "Input source path or YouTube URL cannot be empty."

    src = source.strip()

    # YouTube URL check
    if src.startswith("http://") or src.startswith("https://"):
        youtube_regex = r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$"
        if not re.match(youtube_regex, src, re.IGNORECASE):
            return False, "Invalid URL format. Please provide a valid YouTube URL (e.g. https://www.youtube.com/watch?v=...)."
        return True, ""

    # Local file path check & path traversal prevention
    if ".." in src or "\x00" in src:
        return False, "Invalid file path: path traversal characters are prohibited."

    path = Path(src)
    if not path.exists():
        return False, f"Local file not found: '{src}'. Please check the file path."

    if not path.is_file():
        return False, f"Specified path is a directory, not a file: '{src}'."

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FILE_EXTENSIONS:
        return False, f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}."

    return True, ""


def validate_api_keys(require_sarvam: bool = False) -> tuple[bool, str]:
    """
    Validate presence of required environment API keys.
    Returns (is_valid: bool, error_message: str)
    """
    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not mistral_key or mistral_key == "your_mistral_api_key_here":
        return False, "🔑 Mistral API key is missing or invalid in .env file."

    if require_sarvam:
        sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()
        if not sarvam_key or sarvam_key == "your_sarvam_api_key_here":
            return False, "🔑 Sarvam AI API key is missing in .env file (required for Hinglish translation)."

    return True, ""
