import os 
import re
import uuid
import logging
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.config import (
    CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RETRIEVER_K,
)

logger = logging.getLogger(__name__)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def sanitize_collection_name(name: str = None) -> str:
    """Sanitize collection name to comply with ChromaDB naming rules."""
    if not name:
        return f"session_{uuid.uuid4().hex[:16]}"
    
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', str(name))
    clean = clean.strip('_')
    if len(clean) < 3:
        clean = f"col_{clean}"
    if len(clean) > 63:
        clean = clean[:63].rstrip('_')
    
    # Ensure starts and ends with alphanumeric character
    if not clean[0].isalnum():
        clean = "c" + clean[1:]
    if not clean[-1].isalnum():
        clean = clean[:-1] + "0"
        
    return clean

import functools

@functools.lru_cache(maxsize=1)
def get_embeddings():
    logger.info(f"Loading HuggingFace embeddings model: '{EMBEDDING_MODEL}'")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": 'cpu'}
    )

def build_vector_store(transcript: str, collection_name: str = None) -> Chroma:
    c_name = sanitize_collection_name(collection_name)
    logger.info(f"Building Chroma vector store with collection ID: {c_name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(transcript or "")

    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i, 'collection_id': c_name})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=c_name,
        persist_directory=CHROMA_DIR,
    )
    logger.info(f"Indexed {len(docs)} chunk(s) into Chroma collection '{c_name}'")

    return vector_store

def load_vector_store(collection_name: str = None) -> Chroma:
    c_name = sanitize_collection_name(collection_name or DEFAULT_COLLECTION_NAME)
    logger.info(f"Loading existing Chroma collection '{c_name}'")
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=c_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vector_store

def get_retriever(vector_store: Chroma, k: int = RETRIEVER_K):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k},
    )




