#!/usr/bin/env python3
"""
Build and search a simple vector store using embeddings.

Commands:
  build --text-dir <dir> --output <pkl>    Build pickle index from .txt files
  search --index <pkl> --query "text" [--top-k N]  Search index for query
"""
import os
import argparse
import pickle
import numpy as np
from generate_embeddings import generate_embedding, chunk_text

def build_index(text_dir: str, index_path: str):
    """Build a vector index from text files and save as pickle."""
    records = []
    for fname in os.listdir(text_dir):
        if not fname.lower().endswith('.txt'):
            continue
        path = os.path.join(text_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Chunk text and embed
        parts = chunk_text(text)
        vecs = []
        for part in parts:
            try:
                vecs.append(generate_embedding(part))
            except Exception as e:
                print(f"Warning: failed to embed chunk of {fname}: {e}")
        if not vecs:
            print(f"Skipping {fname}, no embeddings generated.")
            continue
        # Average vectors for document
        doc_vec = np.mean(vecs, axis=0)
        records.append({'file': fname, 'text': text, 'vector': doc_vec})
        print(f"Indexed {fname}")
    # Save to pickle
    with open(index_path, 'wb') as pf:
        pickle.dump(records, pf)
    print(f"Saved index with {len(records)} documents to {index_path}")

def load_index(index_path: str):
    """Load vector index from pickle file."""
    with open(index_path, 'rb') as pf:
        return pickle.load(pf)

def search_index(query: str, records: list, top_k: int = 5):
    """Search the index for the query and return top_k matches."""
    q_vec = generate_embedding(query)
    # Compute cosine similarities
    sims = []
    q_norm = np.linalg.norm(q_vec)
    for rec in records:
        v = rec['vector']
        sim = float(np.dot(q_vec, v) / (q_norm * np.linalg.norm(v)))
        sims.append((sim, rec))
    # Sort by descending similarity
    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[:top_k]

def main():
    parser = argparse.ArgumentParser(description="Vector store build/search")
    subparsers = parser.add_subparsers(dest='command', required=True)

    build = subparsers.add_parser('build', help='Build index from txt files')
    build.add_argument('--text-dir', default='data/text', help='Directory with .txt files')
    build.add_argument('--output', default='data/embeddings.pkl', help='Output pickle file')

    search = subparsers.add_parser('search', help='Search index for a query')
    search.add_argument('--index', default='data/embeddings.pkl', help='Pickle index file')
    search.add_argument('--query', required=True, help='Query text')
    search.add_argument('--top-k', type=int, default=5, help='Number of top results to return')

    args = parser.parse_args()
    if args.command == 'build':
        build_index(args.text_dir, args.output)
    elif args.command == 'search':
        records = load_index(args.index)
        results = search_index(args.query, records, args.top_k)
        for score, rec in results:
            print(f"{rec['file']} (score: {score:.4f})")
            # Optionally print snippet
            snippet = rec['text'][:200].replace('\n', ' ')
            print(f"  {snippet}...")
        if not results:
            print("No results found.")

if __name__ == '__main__':
    main()