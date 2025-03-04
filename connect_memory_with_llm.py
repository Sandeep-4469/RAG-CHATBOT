import os
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Define the data directory containing PDF files
DATA_PATH = "data/"

# 🔹 Step 1: Load PDF Files
def load_pdf_files(data: str):
    """
    Loads all PDF documents from the specified directory.

    Parameters:
    data (str): The path to the directory containing PDF files.

    Returns:
    List[Document]: A list of document objects extracted from PDFs.
    """
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Load PDF documents
documents = load_pdf_files(DATA_PATH)

# 🔹 Step 2: Create Chunks
def create_chunks(extracted_data):
    """
    Splits the extracted text documents into smaller chunks.

    Parameters:
    extracted_data (List[Document]): A list of document objects.

    Returns:
    List[Document]: A list of smaller text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Split text into chunks
text_chunks = create_chunks(documents)

# 🔹 Step 3: Create Vector Embeddings
def get_embedding_model():
    """
    Loads the sentence transformer model for text embedding.

    Returns:
    HuggingFaceEmbeddings: The embedding model instance.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load embedding model
embedding_model = get_embedding_model()

# 🔹 Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

# Create FAISS database from text chunks
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

# 🔹 Step 5: Setup LLM (Mistral with HuggingFace)

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id: str):
    """
    Loads a Hugging Face language model using the given repository ID.

    Parameters:
    huggingface_repo_id (str): The repository ID of the model on Hugging Face.

    Returns:
    HuggingFaceEndpoint: The loaded LLM instance.
    """
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# 🔹 Step 6: Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know; don't try to make up an answer. 
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template: str):
    """
    Creates a custom prompt template for the LLM.

    Parameters:
    custom_prompt_template (str): The prompt template string.

    Returns:
    PromptTemplate: A formatted prompt template object.
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# 🔹 Step 7: Load FAISS Database and Create QA Chain
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# 🔹 Step 8: Get User Query and Generate Response
user_query = input("Write Query Here: ")

response = qa_chain.invoke({'query': user_query})

# 🔹 Step 9: Print the Results
print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])
