import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

@dataclass
class Config:

    DATA_DIR: str = ""
    INDEX_DIR: str = ""

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    TOP_K: int = 5

    EMBEDDING_MODEL: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    EMBEDDING_DEVICE: str = "cpu"

    GROQ_API_KEY: str = field(
        default_factory=lambda: os.getenv(
            "GROQ_API_KEY", ""
        )
    )

    GROQ_MODEL: str = "llama-3.1-8b-instant"

    LOG_LEVEL: str = "INFO"

    def __post_init__(self):

        if not self.DATA_DIR:
            self.DATA_DIR = os.path.join(
                BASE_DIR,
                "data"
            )

        if not self.INDEX_DIR:
            self.INDEX_DIR = os.path.join(
                BASE_DIR,
                "faiss_index"
            )

cfg = Config()