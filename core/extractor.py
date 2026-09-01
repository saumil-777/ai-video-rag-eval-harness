import time
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm import get_llm
from core.retry import call_with_retry, MistralRateLimitError

logger = logging.getLogger(__name__)

# Pacing delay between consecutive extraction calls (seconds)
_INTER_CALL_DELAY = 1.2


def build_chain(system_prompt: str):
    llm = get_llm(temperature=0.2)
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# Extractive fallbacks (no LLM required)
# ---------------------------------------------------------------------------

def _fallback(label: str) -> str:
    """
    Return a clearly-labelled fallback when the LLM is unavailable for extraction.
    Never pretends that an AI result was generated.
    """
    return (
        f"**⚠️ Demo/Fallback — {label} extraction temporarily unavailable.**\n\n"
        "The AI service is temporarily busy (rate limit or high load). "
        "Transcript, summary, and RAG chat remain fully available. "
        "Please try again in a moment or review the Full Transcript directly."
    )


# ---------------------------------------------------------------------------
# Public extraction functions (Fault-tolerant optional enrichment)
# ---------------------------------------------------------------------------

def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    try:
        return call_with_retry(chain.invoke, transcript)
    except Exception as ex:
        logger.warning(f"Action item extraction failed ({ex}) — using fallback so pipeline can continue.")
        return _fallback("Action item")


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    try:
        return call_with_retry(chain.invoke, transcript)
    except Exception as ex:
        logger.warning(f"Key decision extraction failed ({ex}) — using fallback so pipeline can continue.")
        return _fallback("Key decision")


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    try:
        return call_with_retry(chain.invoke, transcript)
    except Exception as ex:
        logger.warning(f"Question extraction failed ({ex}) — using fallback so pipeline can continue.")
        return _fallback("Open question")