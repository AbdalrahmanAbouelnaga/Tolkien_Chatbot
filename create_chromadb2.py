import os
import re
import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from utils.pdf_utils import find_and_load_chapters_from_book,get_book_chapters_using_json


warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=512)
final_docs = []
DATA_PATH = "./data/books"
CHROMA_PATH = "./chroma"

### --- The Lord of the Rings ---
lotr_path = os.path.join(DATA_PATH, "The Lord of the Rings.pdf")
lotr_loader = PyPDFLoader(lotr_path)
lotr_docs = lotr_loader.load()
lotr_chunks = text_splitter.split_documents(lotr_docs)
final_docs.extend(find_and_load_chapters_from_book(text_splitter=text_splitter,book_name="The Hobbit"))

book_sections = {
    "THE FELLOWSHIP OF THE RING": "The Fellowship of the Ring",
    "THE TWO TOWERS": "The Two Towers",
    "THE RETURN OF THE KING": "The Return of the King"
}

chapter_titles_by_book = {
    "The Fellowship of the Ring": {
        "A Long-expected Party",
        "The Shadow of the Past",
        "Three is Company",
        "A Short Cut to Mushrooms",
        "A Conspiracy Unmasked",
        "The Old Forest",
        "In the House of Tom Bombadil",
        "Fog on the Barrow-downs",
        "At the Sign of The Prancing Pony",
        "Strider",
        "A Knife in the Dark",
        "Flight to the Ford",
        "Many Meetings",
        "The Council of Elrond",
        "The Ring Goes South",
        "A Journey in the Dark",
        "The Bridge of Khazad-dûm",
        "Lothlórien",
        "The Mirror of Galadriel",
        "Farewell to Lórien",
        "The Great River",
        "The Breaking of the Fellowship"
    },
    "The Two Towers": {
        "The Departure of Boromir",
        "The Riders of Rohan",
        "The Uruk-hai",
        "Treebeard",
        "The White Rider",
        "The King of the Golden Hall",
        "Helm’s Deep",
        "The Road to Isengard",
        "Flotsam and Jetsam",
        "The Voice of Saruman",
        "The Palantír",
        "The Taming of Sméagol",
        "The Passage of the Marshes",
        "The Black Gate is Closed",
        "Of Herbs and Stewed Rabbit",
        "The Window on the West",
        "The Forbidden Pool",
        "Journey to the Cross-roads",
        "The Stairs of Cirith Ungol",
        "Shelob’s Lair",
        "The Choices of Master Samwise"
    },
    "The Return of the King": {
        "Minas Tirith",
        "The Passing of the Grey Company",
        "The Muster of Rohan",
        "The Siege of Gondor",
        "The Ride of the Rohirrim",
        "The Battle of the Pelennor Fields",
        "The Pyre of Denethor",
        "The Houses of Healing",
        "The Last Debate",
        "The Black Gate Opens",
        "The Tower of Cirith Ungol",
        "The Land of Shadow",
        "Mount Doom",
        "The Field of Cormallen",
        "The Steward and the King",
        "Many Partings",
        "Homeward Bound",
        "The Scouring of the Shire",
        "The Grey Havens"
    }
}

current_lotr_book = "The Fellowship of the Ring"
current_lotr_chapter = "Unknown"

for doc in lotr_chunks:
    lines = doc.page_content.splitlines()
    assigned = False

    for line in lines:
        clean = line.strip()
        upper = clean.upper()

        for section_key, section_val in book_sections.items():
            if section_key in upper:
                current_lotr_book = section_val

        for title in chapter_titles_by_book.get(current_lotr_book, []):
            if clean.lower() == title.lower():
                current_lotr_chapter = title
                assigned = True
                break

        if assigned:
            break

    final_docs.append(Document(
        page_content=doc.page_content,
        metadata={
            "book_name": current_lotr_book,
            "chapter_name": current_lotr_chapter
        }
    ))

print("✅ The Trilogy Done")

final_docs.extend(get_book_chapters_using_json("book_chapters.json",text_splitter))


# nomic-embed-text or mxbai-embed-large
ollama_embeddings_1024 = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(final_docs,ollama_embeddings_1024,persist_directory=CHROMA_PATH,collection_metadata={"hnsw:space": "cosine"})

print("✅ All books indexed and saved")