import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm import get_llm
from core.config import SUMMARIZATION_CHUNK_SIZE, SUMMARIZATION_CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_transcript(transcript: str) -> list:
    if not transcript or not transcript.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARIZATION_CHUNK_SIZE,
        chunk_overlap=SUMMARIZATION_CHUNK_OVERLAP,
    )
    return splitter.split_text(transcript)


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

    logger.info(f"Summarizing transcript across {len(chunks)} chunk(s)")
    if len(chunks) == 1:
        chunk_summaries = [map_chain.invoke({"text": chunks[0]})]
    else:
        try:
            chunk_summaries = map_chain.batch([{"text": chunk} for chunk in chunks])
        except Exception as e:
            logger.warning(f"Batch summarization failed ({e}), falling back to sequential invocation.")
            chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

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
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "Untitled Meeting"

    llm = get_llm(temperature=0.2)

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) |
        ChatPromptTemplate.from_messages([
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

    return title_chain.invoke(transcript[:2000])





