from typing import Dict, Union

from pypdf import PdfReader
import os
import json
def bookmark_dict(
    bookmark_list, reader: PdfReader, use_labels: bool = False,
) -> Dict[Union[str, int], str]:
    """
    Extract all bookmarks as a flat dictionary.

    Args:
        bookmark_list: The reader.outline or a recursive call
        use_labels: If true, use page labels. If False, use page indices.

    Returns:
        A dictionary mapping page labels (or page indices) to their title

    Examples:
        Download the PDF from https://zenodo.org/record/50395 to give it a try
    """
    result = {}
    for item in bookmark_list:
        if isinstance(item, list):
            # recursive call
            result.update(bookmark_dict(item, reader))
        else:
            page_index = reader.get_destination_page_number(item)
            page_label = reader.page_labels[page_index]
            if use_labels:
                result[page_label] = item.title
            else:
                result[page_index] = item.title
    return result

def get_pdf_files_os(directory_path):
    pdf_files = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf") and os.path.isfile(os.path.join(directory_path, filename)):
            pdf_files.append(filename)
    return pdf_files
if __name__ == "__main__":
    root_path = "./data/books"
    pdfs = get_pdf_files_os(root_path) 
    book_chapters = {}
    for book in pdfs:
        reader = PdfReader(os.path.join(root_path, book))
        bms = bookmark_dict(reader.outline, reader, use_labels=True)
        book_chapters[book] = {}
        for page_nb, title in sorted(bms.items(), key=lambda n: f"{str(n[0]):>5}"):
            book_chapters[book][title.replace("\x00","")] = [page_nb,]
        titles = list(book_chapters[book].keys())
        if book not in ["The Lord of the Rings.pdf","The Book of Lost Tales  Part 2.pdf"]:
            for i,title in enumerate(titles):
                if i != len(titles)-1:
                    book_chapters[book][title] = [int(book_chapters[book][title][0]),int(book_chapters[book][titles[i+1]][0])-1]
    with open("book_chapters.json", "w") as f:
        json.dump(book_chapters, f, indent=4)