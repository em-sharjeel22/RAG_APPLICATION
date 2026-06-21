import logging
from typing import List, Optional

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)

from .config import cfg
from .vector_store import similarity_search

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an expert teaching assistant for Machine Learning and Computer Vision.

Rules:
1. Answer ONLY from the provided context.
2. Do not make up information.
3. If the answer is not found in the context, say:
   "This topic isn't covered in the provided materials."
4. Explain concepts clearly and concisely.
5. Use numbered points when appropriate.
6. Mention source document and page number whenever possible.
7. Keep answers educational and beginner friendly.
"""


def build_prompt(
    question: str,
    context_docs: list,
    chat_history: Optional[List] = None,
):
    """
    Build prompt for LLM.
    """

    chat_history = chat_history or []

    context_parts = []

    for idx, doc in enumerate(
        context_docs,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "?"
        )

        context_parts.append(
            f"""
Source {idx}
Document: {source}
Page: {page}

{doc.page_content}
"""
        )

    context_text = "\n\n".join(
        context_parts
    )

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        )
    ]

    # Last 3 conversation turns
    for exchange in chat_history[-3:]:

        if (
            isinstance(exchange, dict)
            and "user" in exchange
            and "assistant" in exchange
        ):
            messages.append(
                HumanMessage(
                    content=exchange["user"]
                )
            )

            messages.append(
                AIMessage(
                    content=exchange["assistant"]
                )
            )

    messages.append(
        HumanMessage(
            content=f"""
Context:

{context_text}

Question:
{question}

Answer using only the context above.
"""
        )
    )

    return messages


def get_answer(
    question: str,
    vector_store,
    llm,
    chat_history: Optional[List] = None,
    top_k: int = None,
):
    """
    Retrieve relevant chunks and generate answer.
    """

    try:

        top_k = top_k or cfg.TOP_K

        docs = similarity_search(
            vector_store=vector_store,
            query=question,
            k=top_k,
        )

        if not docs:
            return (
                "This topic isn't covered "
                "in the provided materials."
            )

        messages = build_prompt(
            question=question,
            context_docs=docs,
            chat_history=chat_history,
        )

        response = llm.invoke(
            messages
        )

        answer = (
            response.content.strip()
        )

        return answer

    except Exception as e:

        log.exception(
            "Retriever error"
        )

        return (
            f"Error generating answer:\n{str(e)}"
        )