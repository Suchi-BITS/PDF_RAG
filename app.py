import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# ---- Streamlit Setup ----
st.set_page_config(page_title="📚 Chat with Your Files", page_icon=":robot_face:", layout="wide")
st.title("📄 WhatsApp-style Chat with your Documents")

# ---- Sidebar ----
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
uploaded_files = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# ---- Session state ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---- Helper Function ----
def load_and_split_file(file_path):
    ext = file_path.split(".")[-1]
    if ext == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type!")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# ---- Process Files ----
if st.sidebar.button("Process Files"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        documents = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                documents.extend(load_and_split_file(file_path))

        if documents:
            st.success(f"Loaded and split {len(documents)} document chunks.")

            # Embedding setup
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Create Chroma WITHOUT persistence (in-memory)
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embedding,
                collection_name="user_docs"  # ensures new session uses fresh collection
            )

            retriever = vectordb.as_retriever()

            # LLM + QA Chain
            llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")
            qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = []
        else:
            st.error("No valid documents found!")

# ---- Chat UI ----
if st.session_state.qa_chain:
    st.subheader("💬 Chat with Your Documents")

    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

    user_query = st.chat_input("Ask your question...")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain({
                "question": user_query,
                "chat_history": st.session_state.chat_history
            })
            bot_response = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(bot_response)

        st.session_state.chat_history.append((user_query, bot_response))
else:
    st.info("Upload and process your documents first!")
