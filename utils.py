from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_split_file(file_path, file_type=None):
    """
    Load and split a document file into chunks.
    
    Args:
        file_path (str): Path to the file
        file_type (str, optional): File extension type. If None, detected from file_path.
    
    Returns:
        list: List of document chunks
    """
    # Determine file extension
    if file_type is None:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().strip('.')
    else:
        ext = file_type.lower().strip('.')

    # Select appropriate loader
    if ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    # Load and split documents
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    
    return split_docs