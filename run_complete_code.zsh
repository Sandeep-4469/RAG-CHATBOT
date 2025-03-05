python3 -m venv env
source env/bin/activate

pip install -r requirements.txt

python3 create_memory_for_llm.py
python3 connect_memory_with_llm.py

export HF_TOKEN=''

streamlit run chatbot.py --server.fileWatcherType=none
