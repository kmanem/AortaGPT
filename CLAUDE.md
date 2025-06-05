# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AortaGPT is a clinical decision support tool for managing patients at risk for Heritable Thoracic Aortic Disease (HTAD). It integrates AI-powered chat with clinical guidelines and genetic data to provide evidence-based recommendations.

## Commands

### Development
```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run aortagpt_app.py

# Prepare data (if needed)
python extract_pdfs.py      # Extract text from PDFs
python generate_embeddings.py  # Generate embeddings for RAG
```

### Linting & Validation
```bash
# No linting configuration found - ask user for preferred linting setup
# No type checking configuration found - ask user for preferred type checker
```

### Environment Setup
- **Required**: Set `OPENAI_API_KEY` environment variable
- **Python**: 3.8+ required (per README)
- **Default Port**: 8501 (Streamlit)

## Architecture

### Core Components
1. **aortagpt_app.py**: Main Streamlit application with clinical decision logic
   - Sidebar for patient inputs (demographics, genetics, measurements, medications)
   - Tab-based UI with Chat and Report sections
   - Session state management for persistence
   - Integration with text interpretation manager

2. **text_interpretation.py**: State machine-based text parsing system
   - **InterpretationState** enum: IDLE → CONFIRMING → PROCESSING → COMPLETED/ERROR
   - **PatientParameters** dataclass: Structured patient data model
   - **TextInterpretationManager**: Handles the entire interpretation workflow
   - Uses OpenAI Responses API (`gpt-4.1-nano`) for structured JSON extraction
   - Inline UI rendering (no popups) for better UX
   - Single rerun strategy to prevent loops

3. **MasterRag.py**: RAG implementation for document search and chat
   - Integrates with OpenAI GPT-4.1
   - Context injection from vector search results
   - Maintains conversation history

4. **vector_store.py**: Core vector database implementation
   - Pickle-based storage for embeddings
   - Document chunking (5000 chars, 500 overlap)
   - Cosine similarity search
   - Builds index from text files in `data/text/`
   - Uses `text-embedding-3-small` model for efficiency

5. **helper_functions.py**: Clinical utilities and API integrations
   - ClinVar API wrapper with rate limiting
   - Gene-specific risk calculations
   - Clinical recommendation generators
   - Kaplan-Meier survival curve plotting (with fix for numerical stability)

### Data Processing Pipeline
1. **PDF Extraction** (`extract_pdfs.py`):
   - Input: PDFs in `data/raw/`
   - Output: Text files in `data/text/`
   - Uses pypdf for text extraction

2. **Embedding Generation** (`generate_embeddings.py`):
   - Input: Text files from `data/text/`
   - Model: OpenAI text-embedding-3-large
   - Output: JSON and CSV files with embeddings
   - Chunks documents and averages embeddings

3. **Vector Search** (`vector_search.py`):
   - Wrapper around vector_store functions
   - Returns top-k results with text snippets

### API Integration Patterns
- **ClinVar/NCBI**: 
  - Batched requests (100 IDs per call)
  - Exponential backoff for rate limiting
  - Results cached for 1 hour using `@st.cache_data`
  
- **OpenAI**:
  - GPT-4.1 for chat completions
  - gpt-4.1-nano with Responses API for text interpretation
  - text-embedding-3-small for document embeddings (optimized for speed)
  - Structured JSON output with strict schema validation

### Clinical Decision Logic
- Gene-specific surgical thresholds and risk factors
- Dynamic recommendation generation based on:
  - Patient demographics and family history
  - Genetic variants (pathogenic/likely pathogenic)
  - Aortic measurements and growth rates
  - Medication status and clinical symptoms
- Red flag alerts for urgent clinical scenarios
- Guideline-based surveillance intervals

## Important Patterns

### State Management
- Extensive use of Streamlit session state
- State machine pattern for text interpretation (prevents rerun loops)
- Persistent chat history across interactions
- Form-based input validation

### Error Handling
- Graceful degradation for API failures
- User-friendly error messages
- Fallback to cached data when available

### Performance Optimizations
- Lazy loading of genetic variants
- Batched API calls to minimize latency
- Pickle-based vector storage for fast retrieval
- Background threading for non-blocking operations
- Vector search moved to "Initialize Chat" to improve text interpretation speed
- Smaller embedding model (text-embedding-3-small) for faster processing

### UI/UX Conventions
- Collapsible sections with st.expander
- Progress indicators for long operations
- Color-coded risk levels (red/orange/yellow/green)
- Interactive Kaplan-Meier plots with matplotlib

## Knowledge Base

Medical guidelines and research papers are stored in `data/raw/`:
- ACC/AHA 2022 Guidelines (`acc_aha_2022.pdf`)
- European HTN Guidelines (`european_htn_guidelines.pdf`)
- Gene Reviews (`gene_reviews.pdf`)
- HTAD Gene Risks (`htad_gene_risks.pdf`)
- HTAD Pathways (`htad_pathways.pdf`)
- MAC Consortium data (`mac_arterial_events.pdf`, `mac_supplement.pdf`)
- Prakash Aorta 2025 (`prakash_aorta_2025.pdf`)

Additional data source:
- Genomic guidance CSV (`genomic_guidance.csv`)

These are processed into embeddings for the RAG system to provide evidence-based responses.