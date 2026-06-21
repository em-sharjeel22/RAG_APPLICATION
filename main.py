# -*- coding: utf-8 -*-

import argparse
import sys
import os

# allow src imports
sys.path.insert(0, os.path.dirname(__file__))

from src.ingest import ingest_documents
from src.retriever import get_answer
from src.connections import get_llm, get_embeddings
from src.vector_store import load_vector_store


# --------------------------------------------------
# Load shared resources (IMPORTANT FIX)
# --------------------------------------------------

def load_resources():
    embeddings = get_embeddings()
    vector_store = load_vector_store(embeddings=embeddings)
    llm = get_llm()
    return vector_store, llm


# --------------------------------------------------
# INGEST
# --------------------------------------------------

def run_ingest():
    print("\n📚 Starting ingestion (PDF → FAISS)...\n")

    ingest_documents(
        data_dir="data",
        index_dir="faiss_index"
    )

    print("\n✅ Ingestion complete!")
    print("👉 Now run: python main.py --chat\n")


# --------------------------------------------------
# SINGLE QUESTION MODE
# --------------------------------------------------

def run_single_question(question: str):
    print(f"\n🔍 Question: {question}\n")

    vector_store, llm = load_resources()

    answer = get_answer(
        question=question,
        vector_store=vector_store,
        llm=llm,
        chat_history=[]
    )

    print(f"\n💬 Answer:\n{answer}\n")


# --------------------------------------------------
# CHAT MODE (CLI)
# --------------------------------------------------

def run_chat():
    print("\n" + "═" * 60)
    print("🧠 RAG CHAT MODE (type 'exit' to quit)")
    print("═" * 60 + "\n")

    vector_store, llm = load_resources()

    chat_history = []

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit", "q"]:
            break

        answer = get_answer(
            question=question,
            vector_store=vector_store,
            llm=llm,
            chat_history=chat_history
        )

        print(f"\nAssistant: {answer}\n")

        chat_history.append({
            "user": question,
            "assistant": answer
        })


# --------------------------------------------------
# MAIN CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG Chatbot CLI"
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--ingest",
        action="store_true",
        help="Build FAISS index from PDFs"
    )

    group.add_argument(
        "--ask",
        type=str,
        help="Ask a single question"
    )

    group.add_argument(
        "--chat",
        action="store_true",
        help="Interactive chat mode"
    )

    args = parser.parse_args()

    if args.ingest:
        run_ingest()

    elif args.ask:
        run_single_question(args.ask)

    elif args.chat:
        run_chat()


if __name__ == "__main__":
    main()