⚡ Knowledge Base RAG Agent (LlamaIndex + Groq + BGE Embeddings)

This project implements a Retrieval-Augmented Generation (RAG) agent that allows users to upload documents and query them conversationally. It is architected for maximum speed and cost-efficiency by leveraging the Groq API for lightning-fast LLM inference and an open-source HuggingFace model for embeddings.
The user interface is built using Streamlit, providing a simple, interactive chat experience.

🚀 Tech Stack Highlights

LLM Inference: Groq
Provides the large language model (LLM) for answering user questions.
Blazing-Fast Speed and low latency inference.

Embedding Model: HuggingFace
Converts documents and queries into numerical vectors.
Cost-Effective and high-quality (BAAI/bge-small-en-v1.5).

RAG Orchestration: LlamaIndex
Manages the entire data indexing, retrieval, and synthesis process.
Unified Framework for complex data pipelines.

Vector Database: ChromaDB
Stores the generated vector embeddings locally.
Persistent, Local Storage for the knowledge base.

Front-End: Streamlit
Builds the simple, interactive web-based chat interface.
Rapid Prototyping and deployment.

⭐ Features and Limitations

✔ Features

Conversational question answering over uploaded documents

Ultra-fast answer generation using Groq LLM API

Automatic embedding and retrieval powered by BGE + LlamaIndex

Local vector database (ChromaDB) for persistent knowledge

Simple and interactive Streamlit chat interface

Supports multiple document uploads

Privacy-first — all storage local

❗ Limitations

Does not support scanned handwritten documents unless OCR is used

Retrieval quality depends on extracted text clarity

Internet browsing is not supported — answers come only from uploaded files

Knowledge base must be refreshed when files are updated

⚙ Setup and Installation

Follow these steps to get the application running on your local machine.

1. Clone the Repository

git clone <your-repository-url>
cd <your-repository-name>


2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

# Create the environment (name it .venv or venv)
python -m venv .venv 

# Activate the environment (Windows)
.\.venv\Scripts\activate

# Activate the environment (macOS/Linux)
source .venv/bin/activate


3. Install Dependencies
Install all the necessary libraries, including the LlamaIndex connectors for Groq and HuggingFace.

pip install -r requirements.txt


(Note: Ensure you have a requirements.txt file containing all necessary packages, e.g., streamlit, llama-index-core, llama-index-llms-groq, llama-index-embeddings-huggingface, llama-index-vector-stores-chroma, pypdf, python-dotenv, sentence-transformers, chromadb).

4. Configure API Keys
The application requires your Groq API key to power the LLM.
Create a file named .env in the root directory of the project.
Add your Groq key to the file:

# Get your API key from Groq Cloud
GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


5. Run the Application
Execute the Streamlit script from your terminal:

streamlit run app.py


The application will open automatically in your browser (usually at http://localhost:8501
).

❓ How to Use

Upload Documents: Use the file uploader on the left sidebar to select one or more PDF documents.

Indexing: The application will automatically process, embed, and index these documents into the persistent chroma_db/ folder using the BGE model.

Query: Once indexing is complete, type your question into the chat input at the bottom.

Response: The agent will retrieve the most relevant information from your documents, use the Groq LLM to generate a fast, concise answer, and stream the response back to you.

🧹 Cleanup and Data Management

The vector database is stored locally for persistence, but you can easily remove it.
To Clear the Knowledge Base: Simply delete the chroma_db folder from the root directory of your project. The application will rebuild it the next time you run it and upload new files.

# From the project root directory
rm -rf chroma_db/  # macOS/Linux
rd /s /q chroma_db # Windows Command Prompt

🚧 Potential Improvements
Support for more document types (CSV, JSON, websites, YouTube transcripts)
Source citation highlighting in responses
Multi-user authentication and cloud vector storage
Visualization UI for document snippets retrieved during RAG
Switchable LLM support (GPT, Claude, Mistral, Gemma)
Memory-aware conversation history for better multi-turn chat
Deployment to Streamlit Cloud / HuggingFace Spaces / Docker
