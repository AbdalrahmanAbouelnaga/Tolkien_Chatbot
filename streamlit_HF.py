import os
import yaml
import warnings
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f,)


load_dotenv()
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)



warnings.filterwarnings('ignore')
CHROMA_PATH = "./chroma"
if "messages" not in st.session_state:
    st.session_state.messages = []
# Page configuration
st.set_page_config(
    page_title="🎯 RAG Chatbot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🎯 RAG Chatbot")
st.markdown("---")
# Initialize Ollama models
# nomic-embed-text or mxbai-embed-large
ollama_embeddings_1024 = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=ollama_embeddings_1024
)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Enter your Query."):
    with st.chat_message("User"):
        st.markdown(query)
    st.session_state.messages.append({"role":"User","content":query})
    with st.spinner("Wait for it...", show_time=True):
        retrieved_info = []
        results = vector_store.similarity_search_with_score(query, k=100)
        print(len(results))
        i = 0
        for res, score in results:
            if score < 0.6: # Filter retrieved info
                print(f"== Retrieved Data Results ==\nScore: {score:3f}\nContent: {res.metadata}\n")
                retrieved_info.append(f"{i})"+res.page_content) # Store content
                i +=1
        final_response = client.chat.completions.create(
            model="Qwen/Qwen3-30B-A3B-Thinking-2507:nebius",
            messages=[
    {"role":"system","content":config["system_prompt"]},
    {"role":"user","content":f"""
        #### **[4. INPUTS]**

        **[USER QUESTION]:**
        `{query}`

        **[CONTEXT]:**
        `{" ".join(retrieved_info)}`

        """
    }
            ]
        )
        # session.append(AIMessage(content=final_response.content))
        st.session_state.messages.append({"role":"ai",
                                           "content":final_response.choices[0].message.content})
        sorted_rag_list = [x.split(")") for x in retrieved_info]
        sorted_rag_object = {}
        for item in sorted_rag_list:
            sorted_rag_object[item[0]] = item[1]
        with st.chat_message("ai"):
            st.markdown(final_response.choices[0].message.content)
