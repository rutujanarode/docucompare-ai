"""
utils/document_processor.py
Extract text from PDFs and chunk into pieces.
"""

import re
from typing import List
import io


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a PDF uploaded via Streamlit."""
    try:
        import fitz  # PyMuPDF
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except ImportError:
        # Fallback: pdfplumber
        try:
            import pdfplumber
            pdf_bytes = uploaded_file.read()
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            return text.strip()
        except ImportError:
            return "[Error: Install PyMuPDF (pip install pymupdf) or pdfplumber to read PDFs]"


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    
    Args:
        text:       Raw document text
        chunk_size: Number of words per chunk
        overlap:    Number of words to overlap between chunks
    
    Returns:
        List of text chunk strings
    """
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    chunks = []
    step = max(1, chunk_size - overlap)
    
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) < 10:          # skip tiny tail chunks
            continue
        chunks.append(" ".join(chunk_words))

    return chunks
