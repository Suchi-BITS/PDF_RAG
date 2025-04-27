import streamlit as st
import tempfile
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from utils import load_and_split_file

# Streamlit app config
st.set_page_config(page_title="FinOps Chatbot", page_icon=":robot_face:", layout="wide")
st.title("FinOps RAG Chatbot")
st.markdown("Upload PDFs, DOCX, or TXT files to chat with your FinOps knowledge base.")

# Sidebar for inputs
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
uploaded_files = st.sidebar.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# Session State
if "documents" not in st.session_state:
    st.session_state.documents = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load Documents
if st.sidebar.button("Load Data"):
    documents = []

    if uploaded_files:
        st.info(f"Loading {len(uploaded_files)} files...")
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            ext = filename.split('.')[-1].lower()

            if ext not in ["pdf", "docx", "txt"]:
                st.error(f"Unsupported file extension: {ext}")
                continue

            # Save uploaded file to a temp directory
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            try:
                # Load and split the file
                docs = load_and_split_file(tmp_file_path, file_type=ext)
                documents.extend(docs)
                # Clean up temp file
                os.unlink(tmp_file_path)
            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")
                continue

        if documents:
            st.success(f"Loaded {len(documents)} document chunks!")

            # Save documents to session
            st.session_state.documents = documents

            # Create vector store with FAISS
            with st.spinner("Setting up embeddings and retriever..."):
                embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectordb = FAISS.from_documents(documents, embedding)
                retriever = vectordb.as_retriever()

            # Setup LLM and QA chain
            if groq_api_key:
                with st.spinner("Setting up LLM and QA chain..."):
                    llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")
                    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
                    st.session_state.qa_chain = qa_chain
            else:
                st.error("Please enter your Groq API Key to enable chat functionality.")
        else:
            st.warning("No documents loaded. Please upload supported files.")
    else:
        st.warning("Please upload files before loading data.")

# Chat Section
if st.session_state.qa_chain:
    st.header("📱 Chat with your documents (WhatsApp Style)")

    # Display chat history
    for user_query, bot_response in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            st.markdown(bot_response)

    user_input = st.chat_input("Ask your FinOps questions...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })
            bot_answer = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(bot_answer)

        st.session_state.chat_history.append((user_input, bot_answer))
else:
    st.info("Upload files and click **Load Data** to start chatting!")