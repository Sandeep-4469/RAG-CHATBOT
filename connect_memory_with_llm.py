import os
from sentence_transformers import SentenceTransformer, util
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
    Load all PDF files from the specified directory.
    
    Input:
        - data (str): Path to the directory containing PDF files.
    Output:
        - List of documents extracted from PDFs.
    """
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

documents = load_pdf_files(DATA_PATH)

# 🔹 Step 2: Create Chunks
def create_chunks(extracted_data):
    """
    Split extracted documents into smaller text chunks for better processing.
    
    Input:
        - extracted_data (list): List of extracted documents.
    Output:
        - List of text chunks with defined size and overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)

text_chunks = create_chunks(documents)

# 🔹 Step 3: Create Vector Embeddings
def get_embedding_model():
    """
    Load a pre-trained Sentence Transformer model for embedding generation.
    
    Input: None
    Output:
        - Pre-trained embedding model instance.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# 🔹 Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

# 🔹 Step 5: Setup LLM
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id: str):
    """
    Load a pre-trained Large Language Model (LLM) from Hugging Face.
    
    Input:
        - huggingface_repo_id (str): Model repository ID from Hugging Face.
    Output:
        - Loaded LLM instance ready for inference.
    """
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.8,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

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
    Define a custom prompt template for the retrieval-augmented LLM.
    
    Input:
        - custom_prompt_template (str): A formatted string defining the prompt.
    Output:
        - PromptTemplate instance.
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# 🔹 Step 7: Load FAISS Database and Create QA Chain
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 7}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# 🔹 Step 8: Find Most Relevant Sentences
def highlight_relevant_text(answer, source_docs):
    """
    Find and highlight the most relevant sentences in the source documents.
    
    Input:
        - answer (str): The generated response from the LLM.
        - source_docs (list): List of documents retrieved as relevant sources.
    Output:
        - String containing relevant document excerpts with highlighted sentences.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    highlighted_texts = []
    
    for doc in source_docs:
        sentences = doc.page_content.split(". ")
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(answer_embedding, sentence_embeddings)
        most_relevant_idx = similarities.argmax().item()
        most_relevant_sentence = sentences[most_relevant_idx]
        highlighted_texts.append(f"**Document:**\n{doc.page_content}\n\n**Most Relevant:** {most_relevant_sentence}\n")
    
    return "\n\n".join(highlighted_texts)

# 🔹 Step 9: Get User Query and Generate Response
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

# 🔹 Step 10: Print the Results
print("\nRESULT:\n", response["result"])
print("\n**Relevant Source Information:**\n", highlight_relevant_text(response["result"], response["source_documents"]))
