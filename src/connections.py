# import os
# from .config import cfg

# def get_llm():
#     from langchain_groq import ChatGroq
#     return ChatGroq(
#     model_name = "llama-3.1-8b-instant", 
#     api_key=os.getenv("GROQ_API_KEY"),
#         temperature=0.0,
#         )

# def get_embeddings():
#     from langchain_huggingface import HuggingFaceEmbeddings
#     return HuggingFaceEmbeddings(
#         model_name=cfg.EMBEDDING_MODEL,
#         model_kwargs={"device": cfg.EMBEDDING_DEVICE},
#         encode_kwargs={"normalize_embeddings": True},
#     )
import logging 
logging.basicConfig(level=logging.INFO)
import os
import streamlit as st  
from functools import lru_cache

from .config import cfg

@st.cache_resource
@lru_cache(maxsize=1)
def get_llm():

    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found."
        )

    return ChatGroq(
        model_name=cfg.GROQ_MODEL,
        api_key=api_key,
        temperature=0.0,
    )

@st.cache_resource
@lru_cache(maxsize=1)
def get_embeddings():

    from langchain_huggingface import (
        HuggingFaceEmbeddings,
    )

    return HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL,
        model_kwargs={
            "device": cfg.EMBEDDING_DEVICE
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )