#!/usr/bin/env python3
"""
Generate embeddings for all .txt documents in the data/text folder.
Saves results as JSON and CSV in the data folder.
"""
import os
import openai
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_embedding(text: str) -> np.ndarray:
    """
    Generate embedding for a given text string.

    Args:
        text: Text to generate embedding for.
    Returns:
        Embedding vector as numpy array.
    """
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return np.array(response.data[0].embedding)
 
def chunk_text(text: str, max_chars: int = 5000, overlap: int = 500) -> list:
    """
    Split text into overlapping chunks not exceeding max_chars characters.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunks.append(text[start:end])
        # Move start forward, keep overlap
        start = end - overlap if end < length else length
    return chunks

def main():
    # Locate text files directory relative to script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    text_dir = os.path.join(base_dir, "data", "text")
    output_dir = os.path.join(base_dir, "data")
    os.makedirs(output_dir, exist_ok=True)

    records = []
    # Iterate through all .txt files
    for fname in os.listdir(text_dir):
        if not fname.lower().endswith('.txt'):
            continue
        path = os.path.join(text_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Embedding: {fname}")
        # Chunk long texts to respect token limits
        texts = chunk_text(text)
        embs = []
        for idx, chunk in enumerate(texts):
            try:
                emb = generate_embedding(chunk)
                embs.append(emb)
            except Exception as e:
                print(f"Failed chunk {idx} of {fname}: {e}")
        if not embs:
            print(f"No embeddings generated for {fname}, skipping.")
            continue
        # Average embeddings to get document-level vector
        doc_emb = np.mean(embs, axis=0)
        records.append({
            'file': fname,
            'embedding': doc_emb.tolist()
        })

    # Save as JSON
    json_path = os.path.join(output_dir, 'embeddings.json')
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(records, jf, indent=2)
    print(f"Saved JSON embeddings to {json_path}")

    # Save as CSV (embedding column as JSON string)
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'embeddings.csv')
    # Serialize embedding lists as JSON strings for CSV
    df['embedding'] = df['embedding'].apply(json.dumps)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV embeddings to {csv_path}")

if __name__ == '__main__':
    main()