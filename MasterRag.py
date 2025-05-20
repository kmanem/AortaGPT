"""Search utilities for RAG pipelines: read and search PDF/CSV files."""

import os
import logging
from pypdf import PdfReader
import csv
import requests
import time
import json
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MasterRAG:
    """
    Utility class providing basic search functions over PDF and CSV documents.
    """

    def __init__(self, data_folder: str = "data/raw/"):
        """
        Initialize with a data folder containing data files.

        Args:
            data_folder: Path to the folder with data files.
        """
        self.data_folder = data_folder
        self.document_cache: Dict[str, Any] = {}
        # Cache for ClinVar API results
        self.clinvar_cache: Dict[str, Any] = {}

    def read_pdf(self, filename: str) -> str:
        """Read and return text content from a PDF file."""
        if filename in self.document_cache:
            return self.document_cache[filename]

        file_path = os.path.join(self.data_folder, filename)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return ""

        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
            self.document_cache[filename] = text
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {filename}: {e}")
            return ""

    def read_csv(self, filename: str) -> List[Dict]:
        """Read and return list of rows from a CSV file."""
        if filename in self.document_cache:
            return self.document_cache[filename]

        file_path = os.path.join(self.data_folder, filename)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return []

        try:
            rows = []
            with open(file_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            self.document_cache[filename] = rows
            return rows
        except Exception as e:
            logger.error(f"Error reading CSV {filename}: {e}")
            return []

    def search_pdf(self, filename: str, search_term: str, max_results: int = 5) -> List[str]:
        """Search for paragraphs containing a term in a PDF file."""
        text = self.read_pdf(filename)
        if not text:
            return []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        matches: List[str] = []
        for para in paragraphs:
            if search_term.lower() in para.lower():
                matches.append(para)
                if len(matches) >= max_results:
                    break
        return matches

    def search_csv(self, filename: str, column: str, search_term: str) -> List[Dict]:
        """Search for rows where a column contains the search term in a CSV file."""
        rows = self.read_csv(filename)
        matches: List[Dict] = []
        for row in rows:
            if column in row and search_term.lower() in row[column].lower():
                matches.append(row)
        return matches

    def search_all_pdfs(self, search_term: str, max_results: int = 10) -> List[Dict]:
        """Search all PDF files in the data folder for a term and return up to max_results."""
        results: List[Dict] = []
        pdf_files = [f for f in os.listdir(self.data_folder) if f.lower().endswith('.pdf')]
        for filename in pdf_files:
            matches = self.search_pdf(filename, search_term, max_results=3)
            for m in matches:
                results.append({'source': filename, 'text': m})
                if len(results) >= max_results:
                    return results
        return results

    def search_clinvar(self, gene: str, variant: str) -> Dict[str, Any]:
        """Search for variant information in ClinVar."""
        cache_key = f"{gene}:{variant}"
        if cache_key in self.clinvar_cache:
            logger.info(f"Using cached ClinVar data for {cache_key}")
            return self.clinvar_cache[cache_key]

        try:
            search_term = f"{gene}[gene] AND {variant}"
            logger.info(f"Searching ClinVar for {search_term}")

            params = {
                "db": "clinvar",
                "term": search_term,
                "retmode": "json",
                "retmax": 5
            }
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            response = requests.get(f"{base_url}esearch.fcgi", params=params)
            response.raise_for_status()
            search_result = response.json()

            id_list = search_result.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                logger.warning(f"No results found for {search_term}")
                result = {
                    "found": False,
                    "gene": gene,
                    "variant": variant,
                    "message": "Variant not found in ClinVar"
                }
                self.clinvar_cache[cache_key] = result
                return result

            fetch_params = {
                "db": "clinvar",
                "id": ",".join(id_list),
                "retmode": "json",
                "rettype": "variation"
            }
            time.sleep(0.3)
            logger.info(f"Fetching details for variant IDs: {id_list}")
            fetch_response = requests.get(f"{base_url}esummary.fcgi", params=fetch_params)
            fetch_response.raise_for_status()
            details = fetch_response.json()

            result = {
                "found": True,
                "gene": gene,
                "variant": variant,
                "clinvar_data": []
            }
            for cid in id_list:
                var_data = details.get("result", {}).get(cid, {})
                clinical_sig = var_data.get("clinical_significance", {}).get("description", "Unknown") if "clinical_significance" in var_data else "Unknown"
                review_status = var_data.get("review_status", "Not provided")
                result["clinvar_data"].append({
                    "id": cid,
                    "clinical_significance": clinical_sig,
                    "review_status": review_status,
                    "last_updated": var_data.get("last_updated", "Unknown")
                })
            self.clinvar_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Error querying ClinVar API: {e}")
            return {
                "found": False,
                "gene": gene,
                "variant": variant,
                "error": str(e),
                "message": "Error querying ClinVar API"
            }
