import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from groq import Groq
from langchain.chains import ConversationalRetrievalChain

# Streamlit App Configuration
st.set_page_config(page_title="FinOps Chatbot", page_icon=":robot_face:", layout="wide")
st.title("FinOps RAG Chatbot")
st.markdown("Upload **PDF, DOCX, or TXT** files and chat with your knowledge base!")

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
#client = Groq(api_key=api_key)

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File Upload
uploaded_files = st.file_uploader(
    "Upload multiple files (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

# Helper function to load and split files
def load_and_split_file(file_path):
    ext = file_path.lower().split('.')[-1]
    if ext == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

# Load documents when button clicked
if st.button("Load Files into Knowledge Base"):
    documents = []
    if uploaded_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                docs = load_and_split_file(file_path)
                documents.extend(docs)

    if documents:
        st.success(f"Loaded {len(documents)} document chunks!")

        # Setup Embedding + Retriever
        with st.spinner("Setting up embeddings and retriever..."):
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectordb = Chroma.from_documents(documents, embedding, persist_directory="./chroma_db")
            retriever = vectordb.as_retriever()

        # Setup LLM and QA chain
        with st.spinner("Connecting to LLM and setting up chat..."):
            llm = ChatGroq(api_key=api_key, model="llama3-8b-8192")
            qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
            st.session_state.qa_chain = qa_chain

    else:
        st.warning("No valid files uploaded. Please upload PDFs, DOCX, or TXTs.")

# ---- Chat UI Section ----

if st.session_state.qa_chain:
    st.header("Chat")

    # Show previous chat
    for user_query, bot_response in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**You:** {user_query}")
        with st.chat_message("assistant"):
            st.markdown(f"**Bot:** {bot_response}")

    # User input
    user_input = st.chat_input("Type your question here...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(f"**You:** {user_input}")

        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })
            bot_answer = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(f"**Bot:** {bot_answer}")

        # Update chat history
        st.session_state.chat_history.append((user_input, bot_answer))

else:
    st.info("Upload your files and click **Load Files into Knowledge Base** to start chatting!")

