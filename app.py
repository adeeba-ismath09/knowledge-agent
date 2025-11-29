import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# --- LlamaIndex Core ---
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# --- Groq LLM ---
from llama_index.llms.groq import Groq

# --- Embeddings (HuggingFace - Free) ---
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Chroma Vector DB ---
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


# Load .env
load_dotenv()

# --- CONFIGURATION ---
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base_collection"

# GROQ API KEY
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

# ----- Configure Global LLM + Embedding -----
Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.1,
    streaming=True
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Text chunking configuration
Settings.text_splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=200,
)


# --- CORE FUNCTIONS ---
@st.cache_resource(show_spinner=False)
def get_chroma_client_and_store():
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


@st.cache_resource(show_spinner="🔍 Indexing documents...")
def index_documents(uploaded_files, vector_store):
    if not uploaded_files:
        return None

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        reader = SimpleDirectoryReader(input_dir=temp_dir)
        documents = reader.load_data()

        st.info(f"📄 Loaded {len(documents)} documents")

        nodes = []
        for doc in documents:
            nodes.extend(Settings.text_splitter.get_nodes_from_documents([doc]))

        st.info(f"🔪 Split into {len(nodes)} chunks")

    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        show_progress=True
    )
    return index


def get_rag_response(index, query):
    if index is None:
        yield "⚠️ Please upload documents first."
        return

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        streaming=True
    )

    response = query_engine.query(query)

    for chunk in response.response_gen:
        yield chunk


# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="⚡ Groq RAG Knowledge Agent")
st.title("⚡ Knowledge Base AI Agent — Powered by Groq")

with st.expander("🔍 How it works (RAG Pipeline)"):
    st.markdown("""
    1. Your document is divided into small chunks (1024 characters each)
    2. Chunks are stored in ChromaDB (vector database)
    3. When you ask a question — only the most relevant chunks are fetched
    4. Groq reads them and generates an accurate answer
    """)

if "uploaded_files_cache" not in st.session_state:
    st.session_state.uploaded_files_cache = []

uploaded_files = st.file_uploader(
    "Upload documents to index in ChromaDB:",
    type=["pdf", "txt", "docx", "pptx", "csv"],   # Updated here
    accept_multiple_files=True
)

vector_store = get_chroma_client_and_store()
current_files = sorted([f.name for f in uploaded_files])

if current_files != st.session_state.uploaded_files_cache:
    st.session_state.uploaded_files_cache = current_files
    index = index_documents(uploaded_files, vector_store)
    st.session_state.index = index
    if uploaded_files:
        st.success(f"📌 Indexed {len(uploaded_files)} document(s)")
else:
    index = st.session_state.get("index", None)

query = st.chat_input("Ask a question about your uploaded documents...")

if query:
    index = st.session_state.get("index", None)

    if index is None:
        with st.chat_message("assistant"):
            st.error("⚠️ Please upload and index documents first.")
    else:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            st.write_stream(get_rag_response(index, query))
