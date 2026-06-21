import streamlit as st
from datetime import datetime

from src.connections import get_llm, get_embeddings
from src.vector_store import load_vector_store
from src.retriever import get_answer
from src.ingest import ingest_documents

# --------------------------------------------------
# Config
# --------------------------------------------------
KNOWLEDGE_BASE = [
    {"name": "Machine Learning Yearning", "author": "Andrew Ng", "pages": 118},
    {"name": "Computer Vision", "author": "Textbook", "pages": 609},
]
TOTAL_CHUNKS = 3509

SAMPLE_QUESTIONS = [
    "What is machine learning?",
    "Explain bias vs variance",
    "What is a convolutional neural network?",
    "How do you split train/dev/test sets?",
]

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { color: #9CA3AF; font-size: 0.95rem; margin-top: 0; margin-bottom: 1.5rem; }
    .kb-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .kb-title { font-weight: 600; font-size: 0.9rem; }
    .kb-meta { color: #9CA3AF; font-size: 0.78rem; }
    .stat-pill {
        display: inline-block;
        background: rgba(124, 92, 255, 0.15);
        border: 1px solid rgba(124, 92, 255, 0.4);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    div[data-testid="stChatMessage"] { border-radius: 12px; padding: 4px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Cached resources — loaded once per session
# --------------------------------------------------
@st.cache_resource
def load_vector_store_cached(_embeddings):
    return load_vector_store(embeddings=_embeddings)

embeddings = get_embeddings()
llm = get_llm()

vector_store = None
load_error = None
try:
    vector_store = load_vector_store_cached(embeddings)
except Exception as e:
    load_error = str(e)


def export_chat_as_markdown(messages):
    lines = [f"# RAG Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    for msg in messages:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        lines.append(f"**{role}:**\n\n{msg['content']}\n")
    return "\n".join(lines)

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "top_k" not in st.session_state:
    st.session_state.top_k = 4
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 RAG Chatbot")
    st.caption("Powered by FAISS · LangChain · Groq")

    if vector_store is not None:
        st.success("🟢 Index loaded and ready")
    else:
        st.error(f"🔴 Index not loaded: {load_error}")

    st.divider()

    st.markdown("### 📚 Knowledge Base")
    for doc in KNOWLEDGE_BASE:
        st.markdown(f"""
        <div class="kb-card">
            <div class="kb-title">{doc['name']}</div>
            <div class="kb-meta">{doc['author']} · {doc['pages']} pages</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f'<span class="stat-pill">🧩 {TOTAL_CHUNKS:,} chunks indexed</span>', unsafe_allow_html=True)

    st.divider()
    st.session_state.top_k = st.slider("Top K chunks", 1, 10, st.session_state.top_k)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Re-ingest", use_container_width=True):
            with st.spinner("Processing PDFs..."):
                try:
                    ingest_documents()
                    st.cache_resource.clear()
                    st.success("Re-ingested! Reloading...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingest failed: {e}")
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    if st.session_state.messages:
        st.download_button(
            "⬇️ Export Chat",
            data=export_chat_as_markdown(st.session_state.messages),
            file_name=f"rag_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.markdown('<p class="main-header">📚 Ask your PDFs</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions grounded in your Machine Learning & Computer Vision references.</p>', unsafe_allow_html=True)

clicked_question = None
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, q in zip(cols, SAMPLE_QUESTIONS):
        with col:
            if st.button(q, use_container_width=True, key=f"sample_{q}"):
                clicked_question = q

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

typed_question = st.chat_input("Ask anything...")
question = clicked_question or typed_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if vector_store is None:
                answer = "❌ Vector store isn't loaded. Try re-ingesting from the sidebar."
            else:
                try:
                    answer = get_answer(
                        question=question,
                        vector_store=vector_store,
                        llm=llm,
                        chat_history=st.session_state.chat_history,
                        top_k=st.session_state.top_k,
                    )
                except Exception as e:
                    answer = f"❌ Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append({"user": question, "assistant": answer})
    st.rerun()