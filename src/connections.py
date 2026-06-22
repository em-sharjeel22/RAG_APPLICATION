import logging
import os
import streamlit as st
from .config import cfg

logging.basicConfig(level=logging.INFO)


@st.cache_resource
def get_llm():
    from langchain_groq import ChatGroq

    # st.secrets for Streamlit Cloud, os.getenv for local
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found.")

    return ChatGroq(
        model_name=cfg.GROQ_MODEL,
        api_key=api_key,
        temperature=0.0,
    )


@st.cache_resource
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL,
        model_kwargs={"device": cfg.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )