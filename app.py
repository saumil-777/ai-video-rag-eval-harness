# -*- coding: utf-8 -*-
import streamlit as st
import os
import time
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question



# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress / Status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--success); }
.dot-pending  { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
}

.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }

.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.1);  border: 1px solid rgba(6,182,212,0.2);   align-self: flex-start; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

import uuid
import logging
from core.config import validate_source_input, validate_api_keys
from utils.audio_processor import process_input, cleanup_audio_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    load_dotenv(override=True)
    is_valid_source, src_err = validate_source_input(source)
    is_valid_key, key_err = validate_api_keys(require_sarvam=(language.lower() == "hinglish"))

    if not is_valid_source:
        st.error(f"⚠️ {src_err}")
    elif not is_valid_key:
        st.error(f"🔑 {key_err}")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        session_id = f"session_{uuid.uuid4().hex[:16]}"
        chunks = []
        main_wav_path = None

        try:
            with st.status("🚀 Processing Pipeline...", expanded=True) as status:
                status.write("🔊 Downloading and converting audio...")
                chunks, main_wav_path = process_input(source)

                status.write(f"📝 Transcribing audio ({len(chunks)} chunk(s)) with Whisper...")
                transcript = transcribe_all(chunks, language)

                status.write("🏷️ Generating meeting title...")
                title = generate_title(transcript)

                status.write("📋 Summarising meeting transcript...")
                summary = summarize(transcript)

                status.write("🔍 Extracting action items, key decisions, and questions...")
                action_items  = extract_action_items(transcript)
                decisions     = extract_key_decisions(transcript)
                questions     = extract_questions(transcript)

                status.write("🧠 Building RAG vector index...")
                rag_chain = build_rag_chain(transcript, collection_name=session_id)

                status.update(label="✅ Analysis complete!", state="complete", expanded=False)

            st.session_state.result = {
                "session_id": session_id,
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            time.sleep(0.5)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
        finally:
            cleanup_audio_files(chunks, main_wav_path)

# ── Main Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎬 Main Assistant", "RAG Evaluation & Observability"])

with tab1:
    # ── Results ──────────────────────────────────────────────────────────────────────
    if st.session_state.result:
        r = st.session_state.result

        # Title banner
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📌 Session Title</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
                {r['title']}
            </div>
        </div>""", unsafe_allow_html=True)

        # Top row: summary + transcript
        col1, col2 = st.columns([3, 2], gap="medium")

        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📋 Summary</div>
                <div class="card-content">{r['summary']}</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            with st.expander("📝 Full Transcript", expanded=False):
                st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)

        # Second row: action items | decisions | questions
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">✅ Action Items</div>
                <div class="card-content">{r['action_items']}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🔑 Key Decisions</div>
                <div class="card-content">{r['key_decisions']}</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">&#10068; Open Questions</div>
                <div class="card-content">{r['open_questions']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Export Options ────────────────────────────────────────────────────────
        from utils.exporter import generate_pdf_report, generate_txt_report

        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:0.5rem">📥 Export Options</div>', unsafe_allow_html=True)
        exp_col1, exp_col2, _ = st.columns([1, 1, 2], gap="small")

        txt_data = generate_txt_report(
            r["title"], r["summary"], r["action_items"], r["key_decisions"], r["open_questions"], r["transcript"]
        )
        pdf_data = generate_pdf_report(
            r["title"], r["summary"], r["action_items"], r["key_decisions"], r["open_questions"], r["transcript"]
        )

        with exp_col1:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_data,
                file_name="meeting_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with exp_col2:
            st.download_button(
                label="📝 Download TXT Report",
                data=txt_data,
                file_name="meeting_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown("---")

        # ── RAG Chat ──────────────────────────────────────────────────────────────
        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

        # Chat history display
        if st.session_state.chat_history:
            chat_html = '<div class="chat-container">'
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    chat_html += f"""
                    <div class="chat-msg" style="align-items:flex-end">
                        <span class="chat-label user-label">You</span>
                        <div class="chat-bubble user-bubble">{msg['content']}</div>
                    </div>"""
                else:
                    chat_html += f"""
                    <div class="chat-msg" style="align-items:flex-start">
                        <span class="chat-label bot-label">&#129302; Assistant</span>
                        <div class="chat-bubble bot-bubble">{msg['content']}</div>
                    </div>"""
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem">
                <div style="font-size:2rem;margin-bottom:0.5rem">&#128172;</div>
                <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
            </div>""", unsafe_allow_html=True)

        # Chat input
        chat_col1, chat_col2 = st.columns([5, 1], gap="small")
        with chat_col1:
            user_input = st.text_input("Your question", placeholder="What were the main decisions made?", label_visibility="collapsed")
        with chat_col2:
            send_btn = st.button("Send →", use_container_width=True)

        if send_btn and user_input.strip():
            with st.spinner("Thinking…"):
                answer = ask_question(r["rag_chain"], user_input.strip())
            st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    else:
        # Empty state
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
            <div style="font-size:4rem;margin-bottom:1rem">&#127916;</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
                Ready to Analyse
            </div>
            <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
                Paste a YouTube URL or local file path in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
            </div>
            <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
                <span class="badge badge-purple">Transcription</span>
                <span class="badge badge-cyan">Summarisation</span>
                <span class="badge badge-green">RAG Chat</span>
            </div>
        </div>""", unsafe_allow_html=True)


with tab2:
    from core.evaluation import (
        get_benchmark_dataset,
        BENCHMARK_DATASET_VERSION,
        run_ragas_evaluation,
        generate_video_qa_pairs,
        EVAL_MODE_BENCHMARK,
        EVAL_MODE_CURRENT_VIDEO,
        load_evaluation_history
    )
    from core.rag_engine import build_rag_chain

    st.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-size:1.4rem;font-weight:700;margin-bottom:0.5rem">'
        '📊 RAG Quality Evaluation & Observability</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="color:var(--text-muted);font-size:0.85rem;margin-bottom:1.5rem">'
        'Empirically evaluate Context Precision, Faithfulness, and Answer Relevancy using Ragas.</div>',
        unsafe_allow_html=True
    )

    # ── Mode Selector ─────────────────────────────────────────────────────────
    has_active_session = bool(
        st.session_state.get("result") and
        "rag_chain" in (st.session_state.result or {})
    )

    st.markdown("**Evaluation Mode:**")
    mode_col1, mode_col2 = st.columns(2, gap="small")

    with mode_col1:
        benchmark_selected = st.button(
            "📋 Benchmark Evaluation",
            use_container_width=True,
            key="btn_mode_benchmark",
            help="Uses the built-in 12-question Kickoff Meeting benchmark. No video required. Used for regression testing."
        )

    with mode_col2:
        video_selected = st.button(
            "🎬 Current Video Evaluation",
            use_container_width=True,
            key="btn_mode_video",
            disabled=not has_active_session,
            help="Requires a video to be processed first. Generates questions from that video's transcript."
        )

    # Track selected mode in session state
    if benchmark_selected:
        st.session_state.eval_mode_selected = EVAL_MODE_BENCHMARK
    elif video_selected and has_active_session:
        st.session_state.eval_mode_selected = EVAL_MODE_CURRENT_VIDEO

    eval_mode = st.session_state.get("eval_mode_selected", EVAL_MODE_BENCHMARK)

    # ── Mode Info Panel ───────────────────────────────────────────────────────
    if eval_mode == EVAL_MODE_BENCHMARK:
        st.info(
            f"**🔬 Benchmark Mode** — Kickoff Meeting Benchmark Dataset "
            f"(v{BENCHMARK_DATASET_VERSION} — 12 Curated QA Pairs). "
            "Uses fixed reference answers. Does NOT require a processed video. "
            "Use this for regression testing after pipeline changes."
        )
    else:
        r = st.session_state.result
        session_title = r.get("title", "Current Session") if r else "Unknown Session"
        transcript_len = len(r.get("transcript", "")) if r else 0
        st.info(
            f"**🎬 Current Video Mode** — Session: **{session_title}** "
            f"({transcript_len:,} chars). "
            "Questions will be auto-generated from this transcript. "
            "Reference answers are grounded in the transcript only. "
            "Benchmark questions are NOT used."
        )

    if not has_active_session and eval_mode == EVAL_MODE_CURRENT_VIDEO:
        st.warning("⚠️ No video has been processed yet. Go to the **🎬 Main Assistant** tab, paste a YouTube URL and click **Analyse**, then return here.")

    # ── Run Button ────────────────────────────────────────────────────────────
    run_eval_btn = st.button("⚡ Run Ragas Evaluation", use_container_width=False, key="run_eval_main")

    if run_eval_btn:
        if eval_mode == EVAL_MODE_CURRENT_VIDEO and not has_active_session:
            st.error("❌ Cannot run Current Video Evaluation: no active session. Process a video first.")
        else:
            with st.spinner("Executing RAG Pipeline & Ragas Evaluation Metrics..."):
                try:
                    if eval_mode == EVAL_MODE_BENCHMARK:
                        benchmark_data = get_benchmark_dataset()
                        chain_to_eval = build_rag_chain(
                            benchmark_data["transcript"],
                            collection_name="eval_benchmark_session"
                        )
                        qa_pairs_to_eval = None  # evaluator loads benchmark internally
                        session_id = "eval_benchmark_session"

                    else:  # EVAL_MODE_CURRENT_VIDEO
                        r = st.session_state.result
                        chain_to_eval = r["rag_chain"]
                        session_id = r.get("session_id", "active_session")
                        transcript = r.get("transcript", "")

                        with st.spinner("Generating evaluation questions from video transcript..."):
                            qa_pairs_to_eval = generate_video_qa_pairs(transcript, num_questions=5)

                        st.caption(f"✅ Generated {len(qa_pairs_to_eval)} evaluation questions from transcript.")

                    latest_run = run_ragas_evaluation(
                        rag_chain=chain_to_eval,
                        qa_pairs=qa_pairs_to_eval,
                        session_id=session_id,
                        save_history=True,
                        evaluation_mode=eval_mode
                    )
                    st.session_state.latest_eval_run = latest_run
                    mode_label = "Benchmark" if eval_mode == EVAL_MODE_BENCHMARK else "Current Video"
                    st.success(
                        f"✅ Ragas {mode_label} Evaluation Complete! Run ID: `{latest_run['run_id']}`"
                    )
                except Exception as ex:
                    st.error(f"❌ Evaluation Error: {ex}")

    # ── Results ───────────────────────────────────────────────────────────────
    history_runs = load_evaluation_history()
    latest_run = st.session_state.get("latest_eval_run") or (history_runs[-1] if history_runs else None)

    if latest_run:
        st.markdown("---")
        run_mode_label = "📋 Benchmark" if latest_run.get("evaluation_mode") == EVAL_MODE_BENCHMARK else "🎬 Current Video"
        st.markdown(f"### 📈 Overall RAG Quality Metrics — {run_mode_label}")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        cp = latest_run.get("context_precision", 0.0)
        f = latest_run.get("faithfulness", 0.0)
        ar = latest_run.get("answer_relevancy", 0.0)
        ov = latest_run.get("overall_score", 0.0)

        cp_delta = latest_run.get("context_precision_delta", 0.0)
        f_delta = latest_run.get("faithfulness_delta", 0.0)
        ar_delta = latest_run.get("answer_relevancy_delta", 0.0)
        ov_delta = latest_run.get("overall_score_delta", 0.0)

        m_col1.metric("Context Precision", f"{cp:.2f}", f"{cp_delta:+.2f}" if cp_delta != 0 else None)
        m_col2.metric("Faithfulness", f"{f:.2f}", f"{f_delta:+.2f}" if f_delta != 0 else None)
        m_col3.metric("Answer Relevancy", f"{ar:.2f}", f"{ar_delta:+.2f}" if ar_delta != 0 else None)
        m_col4.metric("Overall RAG Score", f"{ov:.2f}", f"{ov_delta:+.2f}" if ov_delta != 0 else None)

        # Visual Chart
        st.markdown("#### 📊 Metric Score Breakdown")
        chart_data = {
            "Metric": ["Context Precision", "Faithfulness", "Answer Relevancy", "Overall Score"],
            "Score": [cp, f, ar, ov]
        }
        st.bar_chart(chart_data, x="Metric", y="Score")

        # Per-Question Results
        st.markdown("---")
        st.markdown(f"### 📋 Per-Question Evaluation Breakdown (Run: `{latest_run['run_id']}`)")

        results_list = latest_run.get("per_question_results", [])
        if results_list:
            import pandas as pd
            df_display = pd.DataFrame([
                {
                    "Question": q["question"],
                    "Reference Answer": q["reference_answer"],
                    "Generated Answer": q["generated_answer"],
                    "Context Precision": q["context_precision"],
                    "Faithfulness": q["faithfulness"],
                    "Answer Relevancy": q["answer_relevancy"],
                    "Overall": q["overall_score"],
                }
                for q in results_list
            ])
            st.dataframe(df_display, use_container_width=True)

            with st.expander("🔍 Inspect Retrieved Contexts per Question"):
                for idx, q_res in enumerate(results_list):
                    st.markdown(f"**Q{idx+1}: {q_res['question']}**")
                    st.markdown(f"*Generated Answer*: {q_res['generated_answer']}")
                    st.markdown("*Retrieved Chunks*:")
                    for c_idx, chunk in enumerate(q_res.get("retrieved_contexts", [])):
                        st.code(f"Chunk [{c_idx+1}]: {chunk}")
                    st.markdown("---")

    # History & Regression Comparison
    if history_runs:
        st.markdown("---")
        st.markdown("### 📜 Evaluation Run History & Regression Comparison")

        history_df_data = []
        for run in reversed(history_runs):
            mode_icon = "📋" if run.get("evaluation_mode", "benchmark") == EVAL_MODE_BENCHMARK else "🎬"
            history_df_data.append({
                "Mode": f"{mode_icon} {run.get('evaluation_mode', 'benchmark')}",
                "Run ID": run.get("run_id"),
                "Timestamp": run.get("timestamp", "")[:19],
                "Session": run.get("session_id", "")[:30],
                "Questions": run.get("num_questions"),
                "Context Precision": run.get("context_precision"),
                "Faithfulness": run.get("faithfulness"),
                "Answer Relevancy": run.get("answer_relevancy"),
                "Overall Score": run.get("overall_score"),
                "Overall Δ": f"{run.get('overall_score_delta', 0.0):+.4f}"
            })

        import pandas as pd
        st.dataframe(pd.DataFrame(history_df_data), use_container_width=True)
    else:
        st.info("ℹ️ No historical evaluation runs recorded yet. Click '⚡ Run Ragas Evaluation' to trigger.")

