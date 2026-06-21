import glob
import logging
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import cfg
from .connections import get_embeddings
from .vector_store import save_vector_store

logging.basicConfig(
    level=cfg.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)


def load_pdfs(data_dir=None):
    """
    Load all PDF files from data directory.
    """

    data_dir = data_dir or cfg.DATA_DIR

    pdf_files = glob.glob(
        os.path.join(data_dir, "*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {data_dir}"
        )

    documents = []

    for pdf_path in pdf_files:

        log.info(
            "Loading PDF: %s",
            os.path.basename(pdf_path)
        )

        pages = PyMuPDFLoader(
            pdf_path
        ).load()

        # Clean metadata
        for page in pages:
            page.metadata["source"] = (
                os.path.basename(pdf_path)
            )

        documents.extend(pages)

        log.info(
            "Loaded %d pages",
            len(pages)
        )

    log.info(
        "Total pages loaded: %d",
        len(documents)
    )

    return documents


def chunk_documents(documents):
    """
    Split documents into chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    log.info(
        "Created %d chunks",
        len(chunks)
    )

    return chunks


def ingest_documents(
    data_dir=None,
    index_dir=None
):
    """
    Complete ingestion pipeline:
    PDF -> Chunks -> Embeddings -> FAISS
    """

    data_dir = data_dir or cfg.DATA_DIR
    index_dir = index_dir or cfg.INDEX_DIR

    log.info("Starting ingestion...")

    documents = load_pdfs(
        data_dir=data_dir
    )

    chunks = chunk_documents(
        documents
    )

    embeddings = get_embeddings()

    save_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        index_dir=index_dir
    )

    log.info(
        "Ingestion completed successfully."
    )

    return len(chunks)