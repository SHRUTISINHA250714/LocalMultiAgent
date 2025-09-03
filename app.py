import asyncio
import sys
import tempfile
import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.vectorstores import Chroma
from langchain_milvus import Milvus

from agents.retrieval_agent import async_retrieve_agent
from agents.reasoning_agent import reasoning_agent
from agents.response_agent import response_agent

if sys.version_info >= (3, 8):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chroma_persist_dir = "chroma_db"

st.set_page_config(page_title="Multi-Agentic RAG", layout="wide")


st.markdown(
    """
    <style>
    /* Increase sidebar width */
    .css-1d391kg {width: 350px;}
    /* Style vector DB buttons */
    .vector-btn {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        margin: 4px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .vector-btn:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True
)


st.sidebar.title("Configuration")


vector_db = st.sidebar.radio("Vector DB", ("chroma", "milvus"))

st.sidebar.markdown(f"**Currently using:** {vector_db}")

uploaded_files = st.sidebar.file_uploader(
    "Upload files (multiple)",
    type=["txt", "pdf", "docx", "xlsx"],
    accept_multiple_files=True
)

if st.sidebar.button("Index uploaded files"):
    if not uploaded_files:
        st.sidebar.warning("Please upload files first!")
    else:
        docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            if uploaded_file.name.endswith(".txt"):
                loader = TextLoader(tmp_path)
            elif uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(tmp_path)
            elif uploaded_file.name.endswith(".xlsx"):
                loader = UnstructuredExcelLoader(tmp_path, mode="elements")
            else:
                st.sidebar.error(f"Unsupported file type: {uploaded_file.name}")
                continue

            docs.extend(loader.load())
            os.remove(tmp_path)

        chunks = splitter.split_documents(docs)

        if vector_db == "milvus":
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name="multiagent_docs",
                connection_args={"host": "localhost", "port": "19530"},
            )
            vectorstore.add_documents(chunks)
            st.sidebar.success(f"Embedded {len(chunks)} chunks into Milvus")
        else:
            vectorstore = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)
            vectorstore.add_documents(chunks)
            vectorstore.persist()
            st.sidebar.success(f"Embedded {len(chunks)} chunks into Chroma")



st.title("Multi-Agentic RAG with Ollama")
st.markdown("Upload files, index them, and ask questions using **ChromaDB** or **Milvus** + Ollama.")


col1, col2 = st.columns([1.9, 1])  

with col1:
    st.header("Ask a Question")
    query = st.text_area("Enter your question here:")

    get_answer_clicked = st.button("Get Answer") 

with col2:
    st.image("banner.png", width=550)


if get_answer_clicked:
    if not query:
        st.warning("Please enter a question first!")
    else:
        with st.spinner("Agents are working..."):
            if vector_db == "milvus":
                vectorstore = Milvus(
                    embedding_function=embeddings,
                    collection_name="multiagent_docs",
                    connection_args={"host": "localhost", "port": "19530"},
                )
            else:
                vectorstore = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)

            retrieved_docs = asyncio.run(async_retrieve_agent(vectorstore, query, k=2))
            reasoning = reasoning_agent(retrieved_docs, query)
            final_answer = response_agent(reasoning, query)

        st.subheader("Answer")
        st.write(final_answer)
        with st.expander("Retrieved Context"):
            for i, d in enumerate(retrieved_docs, 1):
                st.markdown(f"**Doc {i}:** {d.page_content[:300]}...")
