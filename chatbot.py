import os
import streamlit as st
import sys
from sentence_transformers import SentenceTransformer, util
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Path to FAISS vector database
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """
    Load the FAISS vector database.
    
    Returns:
        FAISS: Loaded FAISS vector store.
    """
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    """
    Create a custom prompt template for retrieval-based QA.
    
    Args:
        custom_prompt_template (str): The custom template string.
    
    Returns:
        PromptTemplate: Configured prompt template.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    """
    Load the Hugging Face model endpoint for inference.
    
    Args:
        huggingface_repo_id (str): Hugging Face repository ID.
        HF_TOKEN (str): Hugging Face API token.
    
    Returns:
        HuggingFaceEndpoint: Configured model endpoint.
    """
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.8,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def highlight_relevant_text(answer, source_docs):
    """
    Identify and highlight the most relevant sentence from the source documents.
    
    Args:
        answer (str): Generated response.
        source_docs (list): List of source documents used for retrieval.
    
    Returns:
        str: Formatted text highlighting the most relevant sentence from each source document.
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
        
        highlighted_texts.append(f"""
            📄 **Document:**
            {doc.page_content.replace('.', '.\n')}
            
            🔍 **Most Relevant:**
            {most_relevant_sentence}
        """)
    
    return "\n".join(highlighted_texts)

def main():
    """
    Streamlit application for chatbot interaction using retrieval-augmented generation (RAG).
    """
    show_relevant_docs = len(sys.argv) > 1 and sys.argv[1].lower() == 'true'
    
    st.title("Ask Chatbot!")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        role_class = "user-bubble" if message['role'] == 'user' else "assistant-bubble"
        role_icon = "🧑" if message['role'] == 'user' else "🤖"
        st.markdown(f"""
            <div class='chat-container'>
                <div class='chat-bubble {role_class}'>
                    <strong>{role_icon} {message['role'].capitalize()}:</strong><br>
                    {message['content'].replace('.', '.\n')}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    prompt = st.chat_input("Pass your prompt here")
    
    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
        Don't provide anything out of the given context.
        
        Context: {context}
        Question: {question}
        
        Start the answer directly. No small talk, please.
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        
        try:
            vectorstore = get_vectorstore()
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 7}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            
            response = qa_chain.invoke({'query': prompt})
            result = response["result"].replace('.', '.\n')
            source_documents = response["source_documents"]
            
            result_to_show = f"""
                🤖 **Assistant:**
                {result}
            """
            
            if show_relevant_docs:
                result_to_show += "\n---\n**Relevant Source Information:**\n" + highlight_relevant_text(result, source_documents)
            
            st.markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()