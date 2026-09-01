import time
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm import get_llm
from core.config import SUMMARIZATION_CHUNK_SIZE, SUMMARIZATION_CHUNK_OVERLAP
from core.retry import call_with_retry, MistralRateLimitError

logger = logging.getLogger(__name__)

# Pacing delay between sequential LLM calls (seconds).
# Keeps per-minute token throughput within Mistral free-tier limits.
_INTER_CALL_DELAY = 1.2


def split_transcript(transcript: str) -> list:
    if not transcript or not transcript.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARIZATION_CHUNK_SIZE,
        chunk_overlap=SUMMARIZATION_CHUNK_OVERLAP,
    )
    return splitter.split_text(transcript)


# ---------------------------------------------------------------------------
# Extractive fallbacks (no LLM required)
# ---------------------------------------------------------------------------

def _extractive_summary_fallback(transcript: str) -> str:
    """
    Return a simple extractive summary when the LLM is unavailable.
    Takes the first sentence of each 3000-char chunk, capped at 6 bullets.

    Clearly labelled as a fallback so users are never misled.
    """
    chunks = split_transcript(transcript) or [transcript[:3000]]
    bullets = []
    for chunk in chunks[:6]:
        # Take the first sentence (up to the first period/newline)
        sentence = chunk.strip().split(".")[0].strip()
        if sentence:
            bullets.append(f"• {sentence}.")

    body = "\n".join(bullets) if bullets else "• (Transcript content available — see Full Transcript tab.)"
    return (
        "**⚠️ Demo/Fallback Result — AI summarization temporarily unavailable.**\n\n"
        "Key points extracted directly from transcript:\n\n"
        + body
    )


def _fallback_title(transcript: str) -> str:
    """Return a simple extractive title when the LLM is unavailable."""
    words = transcript.strip().split()[:8]
    title = " ".join(words).rstrip(".,;:!?")
    return f"{title}… (Demo)"


# ---------------------------------------------------------------------------
# Main summarize function
# ---------------------------------------------------------------------------

def summarize(transcript: str) -> str:
    if not transcript or not transcript.strip():
        logger.warning("Empty transcript provided to summarize()")
        return "No meeting transcript available to summarize."

    llm = get_llm(temperature=0.3)

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    if not chunks:
        return "No meeting transcript available to summarize."

    logger.info(f"Summarizing transcript across {len(chunks)} chunk(s) — sequential with pacing")

    try:
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary_chunk = call_with_retry(map_chain.invoke, {"text": chunk})
            chunk_summaries.append(summary_chunk)
            # Pace between chunk calls to avoid burst rate-limiting
            if i < len(chunks) - 1:
                time.sleep(_INTER_CALL_DELAY)

        combined = "\n\n".join(chunk_summaries)

        combined_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary in bullet points.",
            ),
            ("human", "{text}"),
        ])
        combined_chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | combined_prompt
            | llm
            | StrOutputParser()
        )

        time.sleep(_INTER_CALL_DELAY)
        return call_with_retry(combined_chain.invoke, combined)

    except MistralRateLimitError:
        logger.warning("Summarization failed due to rate limit — returning extractive fallback.")
        raise  # Let app.py catch and show friendly message + fallback


def generate_title(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "Untitled Meeting"

    llm = get_llm(temperature=0.2)

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    try:
        return call_with_retry(title_chain.invoke, transcript[:2000])
    except MistralRateLimitError:
        logger.warning("Title generation failed due to rate limit — using extractive fallback.")
        return _fallback_title(transcript)
