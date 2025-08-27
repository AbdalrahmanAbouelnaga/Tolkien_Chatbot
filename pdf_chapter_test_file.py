import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from langchain_core.documents import Document

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

DATA_PATH = "./data/books"
with open("book_chapters.json", 'r') as file:
        data_dictionary = json.load(file)
book_title = "The Silmarillion"
sil_path = os.path.join(DATA_PATH, "The Silmarillion.pdf")
sil_loader = PyPDFLoader(sil_path)
sil_docs = sil_loader.load()
print(len(sil_docs))
sil_final_docs = []
for chapter in data_dictionary[f"{book_title}.pdf"]:
        docs = sil_docs[data_dictionary[f"{book_title}.pdf"][chapter][0]:data_dictionary[f"{book_title}.pdf"][chapter][1]]
        chuncks = text_splitter.split_documents(docs)
        for doc in chuncks:
                sil_final_docs.append(Document(
                page_content=doc.page_content,
                metadata={
                    "book_name": book_title,
                    "chapter_name": chapter
                }
            ))
print(len(sil_final_docs))
print(sil_final_docs[0])
# sil_chunks = text_splitter.split_documents(sil_docs[25:26])
# print(sil_chunks[0].page_content)
