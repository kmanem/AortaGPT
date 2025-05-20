"""
Simple interface for vector search over precomputed embeddings.

Provides a single function to search a pickle index for a query.
"""
import os
from typing import List, Dict, Any
import numpy as np
from vector_store import load_index, search_index

# Default index path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX = os.path.join(BASE_DIR, 'data', 'embeddings.pkl')

def search_documents(
    query: str,
    index_path: str = DEFAULT_INDEX,
    top_k: int = 5,
    snippet_length: int = 200
) -> List[Dict[str, Any]]:
    """
    Search the vector index for a given query string.

    Args:
        query: Query text to search.
        index_path: Path to the pickle index file.
        top_k: Number of top results to return.
        snippet_length: Number of characters to include in snippet.
    Returns:
        List of dicts with keys: 'file', 'score', 'snippet'.
    """
    # Load index
    records = load_index(index_path)
    # Perform search
    results = []
    sims = search_index(query, records, top_k)
    for score, rec in sims:
        text = rec.get('text', '')
        # Create a short snippet
        snippet = text.replace('\n', ' ')[:snippet_length]
        results.append({
            'file': rec.get('file'),
            'score': score,
            'snippet': snippet
        })
    return results