from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"

def load_pdf_files(data):
    """
    Loads all PDF files from the specified directory.

    Input:
    - data (str): Path to the directory containing PDF files.

    Output:
    - documents (list): List of extracted PDF documents.
    """
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)

def create_chunks(extracted_data):
    """
    Splits extracted documents into smaller text chunks.

    Input:
    - extracted_data (list): List of document objects.

    Output:
    - text_chunks (list): List of text chunks with defined chunk size and overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)

def get_embedding_model():
    """
    Loads the Hugging Face embedding model for text representation.

    Input:
    - None

    Output:
    - embedding_model (HuggingFaceEmbeddings): Embedding model instance.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

DB_FAISS_PATH = "vectorstore/db_faiss"

db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
