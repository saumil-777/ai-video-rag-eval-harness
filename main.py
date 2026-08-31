import uuid
import logging
from dotenv import load_dotenv
from core.config import validate_source_input, validate_api_keys
from utils.audio_processor import process_input, cleanup_audio_files
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

def run_pipeline(source: str, language: str = "english") -> dict:
    is_valid_source, src_err = validate_source_input(source)
    if not is_valid_source:
        raise ValueError(src_err)

    is_valid_key, key_err = validate_api_keys(require_sarvam=(language.lower() == "hinglish"))
    if not is_valid_key:
        raise ValueError(key_err)

    logger.info("Starting AI Video Assistant pipeline...")
    session_id = f"session_{uuid.uuid4().hex[:16]}"
    chunks = []
    main_wav_path = None

    try:
        chunks, main_wav_path = process_input(source)

        transcript = transcribe_all(chunks, language)
        print(f"Raw transcription (first 300 characters): {transcript[:300]}")

        title = generate_title(transcript)
        summary = summarize(transcript)

        action_item = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)

        rag_chain = build_rag_chain(transcript, collection_name=session_id)

        return {
            "session_id": session_id,
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_item,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
        }
    finally:
        cleanup_audio_files(chunks, main_wav_path)

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")