# Text Extraction & Chunking 

import re
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter



# Together, we are using  two modes:
#extract_text_pages_pypdf → fast, for digital PDFs
#extract_text_pages →  OCR, for scanned PDFs

#for clean_text function
"""
Cleans raw text by normalizing whitespace.

- Replaces multiple spaces, tabs, and newlines with a single space.
- Strips leading and trailing whitespace.

Args:
    text (str): Raw extracted text.

Returns:
    str: Cleaned, whitespace-normalized text.
"""


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# for the extract_text_pages_pypdf function
# if text-based pdf : use pypdf 
"""
Extracts text from a PDF using PyPDF2 (for digital/text-based PDFs).

- Reads each page of the PDF and extracts text directly.
- Cleans the text using `clean_text`.
- Skips empty pages.

Args:
    pdf_path (str): Path to the PDF file.
    display_name (str): Human-readable name of the PDF (used in metadata).

Returns:
    list[dict]: List of page dictionaries, each containing:
        - "page" (int): Page number
        - "text" (str): Extracted text
        - "pdf_name" (str): Display name of the PDF
"""


def extract_text_pages_pypdf(pdf_path, display_name):
    pages = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages, start=1):
            txt = clean_text(page.extract_text() or "")
            if txt:
                pages.append({
                    "page": i,
                    "text": txt,
                    "pdf_name": display_name
                })

                # Example structure:
                # {
                #   "page": 1,
                #   "text": "Paracetamol reduces fever...",
                #   "pdf_name": "drug_info.pdf"
                # }

        return pages
    except Exception as e:
        st.error(f"PyPDF extraction error: {e}")
        return []

# if image based scanned pdf : use pytessarct    
# if image : use extract_text_pages function   
"""
Extracts text from a scanned PDF using OCR (Tesseract).

- Converts each PDF page into an image.
- Runs OCR on the image to extract text.
- Cleans the text using `clean_text`.
- Skips empty pages.

Args:
    pdf_path (str): Path to the scanned PDF file.

Returns:
    list[dict]: List of page dictionaries, each containing:
        - "page" (int): Page number
        - "text" (str): OCR-extracted text
        - "pdf_name" (str): Source PDF file name
"""


def extract_text_pages(pdf_path) -> list:
    """
    OCR the PDF and return a list of dicts:
    [{"page": 1, "text": "....", "pdf_name": "file.pdf"}, ...]
    """
    pages = []
    try:
        images = convert_from_path(pdf_path)
        for i, img in enumerate(images, start=1):
            txt = clean_text(pytesseract.image_to_string(img))
            if txt:
                pages.append({"page": i, "text": txt, "pdf_name": pdf_path})
        return pages
    except Exception as e:
        st.error(f"OCR extraction error: {e}")
        return []

# chunking function
"""
Splits a page’s text into smaller chunks (default: 800 characters each).

Why:
- Prevents exceeding token limits in embeddings or LLM input.
- Improves retrieval accuracy by keeping context focused.

Returns:
- texts: list of text chunks (str)
- metas: list of metadata dicts aligned with each chunk
         (includes pdf_name, page number, and chunk_id for citation tracking)
"""

def chunk_page_text(page_dict, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = splitter.split_text(page_dict["text"])
    texts, metas = [], []
    for j, ch in enumerate(chunks):
        texts.append(ch)
        metas.append({
            "pdf_name": page_dict.get("pdf_name"),
            "page": page_dict.get("page"),
            "chunk_id": j
        })
    return texts, metas