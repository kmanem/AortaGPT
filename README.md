# AortaGPT: Clinical Decision Support Tool

AortaGPT is a web-based clinical decision support application designed to assist healthcare providers in managing patients at risk for aortic disease. By integrating patient demographics, genetic variant data from ClinVar, anatomical measurements, and clinical history, the tool generates personalized recommendations and actionable insights.

## Key Features
- Risk stratification with guideline references
- Surgical threshold guidance based on gene-specific criteria
- Imaging surveillance schedules (MRI/CTA recommendations)
- Lifestyle & activity guidelines
- Genetic counseling advice and inheritance patterns
- Red flag alerts for urgent clinical signs
- Kaplan–Meier survival curves comparing patient risk to general population
- References & resource links to primary guidelines and databases

## Getting Started

### Prerequisites
- Python 3.8 or later
- pip (Python package installer)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Launch the Streamlit app:
```bash
streamlit run aortagpt_app.py
```
The app will open in your default browser at http://localhost:8501.

## Software Details

### Architecture & File Overview
- **aortagpt_app.py**: Main Streamlit interface, handling user inputs, session state, and layout.
- **functions.py**: Core utilities for:
  - ClinVar API integration (NCBI E-utilities: esearch, esummary)
  - Rate-limited network calls with exponential backoff
  - Data caching via `@st.cache_data` (TTL = 1 hour)
  - Input validation and risk calculation logic
  - Rendering functions for risk boxes, surgical thresholds, surveillance, lifestyle, counseling, alerts, and Kaplan–Meier plots
- **requirements.txt**: Python dependencies list.
- **README.md**: Project overview and instructions.
- **pyproject.toml** & **uv.lock**: Project metadata and lockfile

### Data Source & Integration
- ClinVar variant fetching uses NCBI E-utilities:
  - `esearch.fcgi` to retrieve variant IDs for pathogenic/likely pathogenic filters
  - `esummary.fcgi` batched requests (default 100 IDs per call) for variant names and details

### Caching Strategy
- All ClinVar API responses are cached with Streamlit's `@st.cache_data` decorator (TTL = 3600 seconds) to reduce redundant network traffic.

### Performance Optimizations
- Batched API calls to minimize HTTP round-trips
- Streamlit data caching to speed up repeated queries
- Lazy loading of heavy computations and plots when triggered by user actions

### Styling & Theming
- Custom CSS included within the Streamlit app for consistent color schemes and interactive elements

### Dependencies
```text
streamlit
requests
matplotlib
numpy
``` 

## Disclaimer
This tool provides informational guidance only and does not replace clinical judgment. All recommendations should be reviewed by qualified healthcare professionals familiar with each patient's unique history and circumstances.
