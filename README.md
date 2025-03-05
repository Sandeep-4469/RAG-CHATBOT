# RAG-CHATBOT

python3 -m venv env
source env/bin/activate
pip install langchain langchain_community langchain_huggingface faiss-cpu pypdf huggingface_hub streamlit
huggingface-cli login
python3 create_memory_for_llm.py
python3 connect_memory_with_llm.py
streamlit run chatbot.py --server.fileWatcherType=none

