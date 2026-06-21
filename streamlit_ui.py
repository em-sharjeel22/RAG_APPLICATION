import streamlit as st
import requests
from datetime import datetime
import os

# --------------------------------------------------
# Config
# --------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

KNOWLEDGE_BASE = [
    {"name": "Machine Learning Yearning", "author": "Andrew Ng", "pages": 118},
    {"name": "Computer Vision", "author": "Textbook", "pages": 609},
]
TOTAL_CHUNKS = 3509  # update if you re-ingest with different chunking

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

# --------------------------------------------------
# Custom CSS — professional look
# --------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .kb-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .kb-title {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .kb-meta {
        color: #9CA3AF;
        font-size: 0.78rem;
    }
    .stat-pill {
        display: inline-block;
        background: rgba(124, 92, 255, 0.15);
        border: 1px solid rgba(124, 92, 255, 0.4);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# API helpers
# --------------------------------------------------
def ask_api(question, top_k):
    try:
        r = requests.post(
            f"{API_URL}/chat",
            json={"question": question, "top_k": top_k},
            timeout=60
        )
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def trigger_ingest():
    try:
        r = requests.post(f"{API_URL}/ingest", timeout=300)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

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
if "last_checked" not in st.session_state:
    st.session_state.last_checked = None

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 RAG Chatbot")
    st.caption("Powered by FAISS · LangChain · Groq")

    status = check_health()
    st.session_state.last_checked = datetime.now().strftime("%H:%M:%S")

    if status:
        st.success(f"🟢 API Online · checked {st.session_state.last_checked}")
    else:
        st.error(f"🔴 API Offline · checked {st.session_state.last_checked}")

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
            with st.spinner("Processing..."):
                res = trigger_ingest()
                st.write(res)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
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

# Sample question chips — only show before first message
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    clicked_question = None
    for col, q in zip(cols, SAMPLE_QUESTIONS):
        with col:
            if st.button(q, use_container_width=True, key=f"sample_{q}"):
                clicked_question = q
else:
    clicked_question = None

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# Input
# --------------------------------------------------
typed_question = st.chat_input("Ask anything...")
question = clicked_question or typed_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result, error = ask_api(question, st.session_state.top_k)
            if error:
                answer = f"❌ Error: {error}"
            else:
                answer = result["answer"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()