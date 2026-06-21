import os
import logging
from langchain_community.vectorstores import FAISS

from .config import cfg

log = logging.getLogger(__name__)


def save_vector_store(chunks, embeddings, index_dir=None):
    """
    Create and save FAISS vector store.
    """

    index_dir = index_dir or cfg.INDEX_DIR

    if not chunks:
        raise ValueError("No chunks provided for indexing.")

    log.info("Building FAISS index from %d chunks...", len(chunks))

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs(index_dir, exist_ok=True)

    vector_store.save_local(index_dir)

    log.info("FAISS index saved successfully.")

    return vector_store


def load_vector_store(index_dir=None, embeddings=None):
    """
    Load existing FAISS vector store.
    """

    index_dir = index_dir or cfg.INDEX_DIR

    if embeddings is None:
        raise ValueError(
            "Embeddings object is required."
        )

    if not os.path.exists(index_dir):
        raise FileNotFoundError(
            f"FAISS index not found: {index_dir}"
        )

    log.info("Loading FAISS index...")

    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )


def similarity_search(
    vector_store,
    query,
    k=None
):
    """
    Retrieve top matching chunks.
    """

    k = k or cfg.TOP_K

    docs = vector_store.similarity_search(
        query,
        k=k
    )

    log.info(
        "Retrieved %d chunks",
        len(docs)
    )

    return docs