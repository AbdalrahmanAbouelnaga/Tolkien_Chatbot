## Tolkien_Chatbot
A RAG chabot using HuggingFace, Qwen3, Chromadb and streamlit
## Install
Clone the repo using:
```
git clone git@github.com:AbdalrahmanAbouelnaga/Tolkien_Chatbot.git
```
## Usage
run create_chromadb.py file to create the Database:
```
python ./create_chromadb.py
```
create a .env file and place your HuggingFace token as such:
```
HF_TOKEN=<Your_HuggingFace_Token>
```
then launch streamlit_HF.py:
```
streamlit run streamlit_HF.py
```
