import whisper
import os
import requests
import logging
from pydub import AudioSegment

from core.config import (
    WHISPER_MODEL,
    SARVAM_STT_MODEL,
    SARVAM_PIECE_SECONDS,
    SARVAM_STT_TRANSLATE_URL,
)

logger = logging.getLogger(__name__)

_model = None


def load_model():
    global _model  
    if _model is None: 
        logger.info(f"Loading local Whisper model: '{WHISPER_MODEL}'...")
        _model = whisper.load_model(WHISPER_MODEL) 
        logger.info("Local Whisper model successfully loaded into memory.")
    return _model 


def transcribe_chunk_whisper(chunk_path: str) -> str:
    if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
        logger.warning(f"Whisper chunk file missing or empty: {chunk_path}")
        return ""
    try:
        audio = AudioSegment.from_file(chunk_path)
        if len(audio) < 500:  # Less than 0.5s
            return ""
    except Exception as e:
        logger.warning(f"Error inspecting audio chunk {chunk_path}: {e}")
        return ""

    model = load_model()  
    result = model.transcribe(chunk_path, task="transcribe")  
    return result.get("text", "")  


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    sarvam_api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not sarvam_api_key or sarvam_api_key == "your_sarvam_api_key_here":
        raise RuntimeError("🔑 SARVAM_API_KEY is not set or invalid in environment/.env file.")

    headers = {"api-subscription-key": sarvam_api_key}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_STT_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        logger.error(f"Sarvam API returned HTTP status {response.status_code}: {response.text}")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            logger.info(f"Sending Sarvam piece {i + 1}/{total_pieces}...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                try:
                    os.remove(piece_path)
                except Exception:
                    pass

    return full_text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = "" 
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    logger.info(f"Using {engine} engine for transcription across {len(chunks)} chunk(s).")

    for i, chunk in enumerate(chunks):  
        logger.info(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)  
        full_transcript += text + " "  

    logger.info("Transcription complete.")
    return full_transcript.strip()
  
