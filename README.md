# 🎬 AI Video & Meeting Assistant

An end-to-end local & cloud AI assistant that ingests YouTube URLs or local video/audio files (`.mp4`, `.wav`, `.mp3`, `.mkv`, `.m4a`), transcribes audio, generates structured meeting summaries, extracts action items, key decisions, and open questions, and enables interactive Retrieval-Augmented Generation (RAG) Q&A over video transcripts.

---

## 🌟 Key Features

- **Audio Extraction & Processing**: Automatic YouTube downloading (`yt-dlp`) or local media conversion to mono 16kHz WAV format with automatic 10-minute chunking.
- **Multilingual Transcription**: Local Speech-to-Text via OpenAI Whisper (`base` model) or Hinglish-to-English translation via Sarvam AI API.
- **Parallel Map-Reduce Summarization**: Fast multi-chunk summarization accelerated with Mistral AI.
- **Structured Extraction**: Automated identification of task owners, action items, key decisions, and unresolved questions.
- **Session-Isolated RAG Engine**: Vector store indexing backed by ChromaDB and HuggingFace embeddings (`all-MiniLM-L6-v2`) with complete per-session data isolation.
- **Prompt Injection Defense**: Security-hardened system prompts to prevent malicious transcript content from overriding system behavior.
- **Export Options**: One-click PDF report generation (`fpdf2`) and plain text export.
- **Clean Streamlit UI & CLI**: Glassmorphic Streamlit interface with sidebar controls and full CLI entry point.

---

## 📐 Architecture Overview

```text
Input (YouTube URL / Media File)
       │
       ▼
Audio Processor (WAV conversion & 10-min chunking)
       │
       ▼
Transcriber (Local Whisper / Sarvam AI API)
       │
       ▼
Centralized LLM Factory (Mistral AI)
 ┌─────┴──────────────────┬──────────────────────┐
 ▼                        ▼                      ▼
Map-Reduce Summarizer   Structured Extractor   ChromaDB Vector Store
(Parallel chunking)     (Action Items/Decisions)  (Session Collection)
 └─────┬──────────────────┴──────────────────────┘
       ▼
Streamlit Dashboard / CLI Interface (RAG Chatbot + PDF/TXT Exports)
```

---

## 🛠️ Prerequisites & Installation

### 1. System Requirements
- **Python**: 3.10 or higher (Python 3.13 supported).
- **FFmpeg**: System binary must be installed and present on your system `PATH`.
  - Windows (winget): `winget install FFmpeg`
  - Linux (apt): `sudo apt install ffmpeg`
  - macOS (brew): `brew install ffmpeg`

### 2. Environment Setup

Clone the repository and install Python dependencies:

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. API Key Configuration

Copy `.env.example` to `.env` and enter your API keys:

```powershell
cp .env.example .env
```

Edit `.env`:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here # Optional (only if transcribing Hinglish)
```

---

## 🚀 Running the Application

### Option A: Streamlit Web UI (Recommended)
Launch the interactive web application:

```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Option B: Command Line Interface (CLI)
Run the pipeline directly from your terminal:

```powershell
python main.py
```

---

## 🧪 Running Automated Tests

Run the full automated test suite (22 unit, integration, and regression tests):

```powershell
python -m unittest discover -s tests
```

Tests verify:
- Centralized configuration and input validation.
- Audio chunking and temporary file cleanup.
- ChromaDB collection sanitization and vector indexing.
- Parallel Map-Reduce transcript summarization.
- Plain text and PDF document generation.
- Per-session vector isolation and failure-safe cleanup regressions.

---

## 📄 License
MIT License. Free for commercial and non-commercial use.
