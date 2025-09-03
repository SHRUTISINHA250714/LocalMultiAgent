# Multi-Agentic RAG with Ollama (Local Setup)

A **Retrieval-Augmented Generation (RAG)** system using multiple **Ollama models** as agents to answer questions based on uploaded documents. This project demonstrates a **local multi-agent AI workflow**, combining document retrieval, reasoning, and response generation in a modular pipeline.

---

## Project Overview

This system allows users to:

1. **Upload documents** (`.txt`, `.pdf`, `.docx`, `.xlsx`) and embed them into a **vector database** (Chroma or Milvus).
2. **Retrieve relevant documents** for a user query using an asynchronous retrieval agent.
3. **Generate reasoning** using a dedicated **Ollama reasoning agent**.
4. **Produce a final answer** with another **Ollama response agent** based on the reasoning output.

The architecture mimics a **multi-agent workflow**, where each agent has a specialized role:

| Agent           | Role                                                             |
| --------------- | ---------------------------------------------------------------- |
| Retrieval Agent | Fetches top-k relevant document chunks from Milvus.              |
| Reasoning Agent | Analyzes retrieved documents to generate structured reasoning.   |
| Response Agent  | Produces a concise and coherent final answer based on reasoning. |

---

## Features

- **Supports multiple file formats**: `.txt`, `.pdf`, `.docx`, `.xlsx`
- **Asynchronous document retrieval** for optimized performance
- **Separate reasoning and response agents** for modularity
- **Embeddings powered by HuggingFace `all-MiniLM-L6-v2`**
- **Vector databases supported:**
  - **Milvus** – efficient similarity search for large datasets
  - **Chroma** – lightweight local option with persistence
- **Streamlit frontend** for interactive queries
- **Temporary file handling** for secure uploads

---

## Tech Stack

- **Python 3.8+**
- **Streamlit** – Frontend interface
- **LangChain + HuggingFace Embeddings** – Document embedding and splitting
- **Milvus / Chroma** – Vector database for semantic search
- **Ollama** – LLaMA-based reasoning and response agents
- **Asyncio** – Optimized asynchronous retrieval

---

## Installation & Setup

Follow these steps to set up and run the Multi-Agentic RAG application locally.

# 1. Clone the repository

git clone https://github.com/SHRUTISINHA250714/LocalMultiAgent.git

cd LocalMultiAgent

# 2. Create a Python virtual environment

python -m venv venv

# 3. Activate the virtual environment

# On Linux/macOS:

source venv/bin/activate

# On Windows:

venv\Scripts\activate

# 4. Install dependencies

pip install -r requirements.txt

# 5. Start Milvus locally (using Docker)

docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.3.0-standalone

# 6. Pull Ollama LLaMA model

ollama pull  llama3.2:1b

# 7. Run the Streamlit app

streamlit run app.py
