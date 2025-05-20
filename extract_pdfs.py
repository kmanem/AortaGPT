#!/usr/bin/env python3
"""
Script to extract text from all PDF files in the data/raw folder
and save them as .txt files in a sibling 'text' directory.
"""
import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a single PDF file."""
    reader = PdfReader(pdf_path)
    text_chunks = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_chunks.append(page_text)
    # Join pages with double newline
    return "\n\n".join(text_chunks)

def main():
    # Define input and output directories
    raw_dir = '/Users/aindukur/Documents/Projects/Personal/AortaGPT/data/raw'
    data_dir = os.path.dirname(raw_dir)
    out_dir = os.path.join(data_dir, 'text')
    os.makedirs(out_dir, exist_ok=True)

    # Process each PDF in the raw directory
    for filename in os.listdir(raw_dir):
        if not filename.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(raw_dir, filename)
        txt = extract_text_from_pdf(pdf_path)
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(out_dir, base_name + '.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(txt)
        print(f"Extracted {filename} -> {os.path.basename(out_path)}")

if __name__ == '__main__':
    main()