import json
import os
import yaml
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import re

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f,)
DATA_PATH = config.get("DATA_PATH")

def find_and_load_chapters_from_book(text_splitter,book_name):
    ## The Hobbit.pdf does not have bookmarks 
    returned_docs = []
    print(DATA_PATH)
    book_path = os.path.join(DATA_PATH, "The Hobbit.pdf")
    book_loader = PyPDFLoader(book_path)
    book_docs = book_loader.load()
    book_chunks = text_splitter.split_documents(book_docs)

    book_chapter_number = "Unknown"
    book_chapter_name = "Unknown"

    for doc in book_chunks:
        text = doc.page_content
        lines = text.splitlines()

        chapter_number = "Unknown"
        chapter_name = "Unknown"

        for i, line in enumerate(lines):
            clean = line.strip()
            match = re.match(r"(?i)^chapter\s+([ivxlcdm\d]+)", clean)
            if match:
                chapter_number = f"Chapter {match.group(1).upper()}"
                for j in range(i + 1, min(i + 6, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not next_line.lower().startswith("chapter"):
                        chapter_name = next_line.title() if next_line.isupper() else next_line
                        break
                book_chapter_number = chapter_number
                book_chapter_name = chapter_name
                break

        if chapter_number == "Unknown":
            chapter_number = book_chapter_number
            chapter_name = book_chapter_name

        returned_docs.append(Document(
            page_content=text,
            metadata={
                "book_name": book_name,
                "chapter_number": chapter_number,
                "chapter_name": chapter_name
            }
        ))

    print(f"✅ {book_name} Done")
    return returned_docs




def get_book_chapters_using_json(json_filename,text_splitter):
    returned_docs = []
    with open(json_filename, 'r') as file:
            data_dictionary = json.load(file)

    for book_title in data_dictionary.keys():
        chapters = data_dictionary[book_title]
        path = os.path.join(DATA_PATH, book_title)
        loader = PyPDFLoader(path)
        docs = loader.load()
        for chapter in chapters.keys():
            chapter_docs = docs[chapters[chapter][0]:chapters[chapter][1]]
            chuncks = text_splitter.split_documents(chapter_docs)
            for doc in chuncks:
                    returned_docs.append(Document(
                    page_content=doc.page_content,
                    metadata={
                        "book_name": book_title.replace(".pdf",""),
                        "chapter_name": chapter
                    }
                ))
        print(f"✅ {book_title} Done")
    return returned_docs