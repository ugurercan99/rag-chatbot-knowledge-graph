import os
import streamlit as st

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'chroma_db')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Copilot",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
    <style>
    .stChatMessage { border-radius: 12px; }
    .source-box {
        background: #1e2d3d;
        border-left: 3px solid #02C39A;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85em;
        color: #8EAAB5;
        margin-top: 6px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 NLP Copilot")
st.caption("A Citation-Grounded Course Chatbot using RAG · B9AI006 NLP CA2")

# ── Load chain (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_chain():
    provider = (os.getenv("LLM_PROVIDER") or "anthropic").lower()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=OUTPUT_DIR,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    if provider == "anthropic" and anthropic_key:
        llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            temperature=0,
            anthropic_api_key=anthropic_key,
        )
    else:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            openai_api_key=openai_key,
        )

    system_prompt = (
        "You are a helpful AI assistant for an MSc NLP course. "
        "Answer only using the retrieved context below. "
        "If the answer is not supported by the context, say clearly that you do not know "
        "based on the provided materials. "
        "Keep the answer concise, structured, and academically clear.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(
                "<div class='source-box'>📌 <b>Sources:</b> " +
                " &nbsp;|&nbsp; ".join(msg["sources"]) +
                "</div>",
                unsafe_allow_html=True
            )

# ── Input ─────────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a question about the NLP course..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chain = load_chain()
                response = chain.invoke({"input": query})
                answer = response["answer"]
                docs = response.get("context", [])

                sources = []
                seen = set()
                for doc in docs:
                    name = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page")
                    label = f"{name} (p.{page+1})" if isinstance(page, int) else name
                    if label not in seen:
                        seen.add(label)
                        sources.append(label)

                st.markdown(answer)
                if sources:
                    st.markdown(
                        "<div class='source-box'>📌 <b>Sources:</b> " +
                        " &nbsp;|&nbsp; ".join(sources) +
                        "</div>",
                        unsafe_allow_html=True
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}")
