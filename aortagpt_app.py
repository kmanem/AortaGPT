import streamlit as st
import openai
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from smolagents import LiteLLMModel  # smolagents imports retained for future use
import threading
from helper_functions import *
from vector_search import search_documents
from text_interpretation import TextInterpretationManager
from report_generator import ReportGenerator
from chat_prompt import chat_system_prompt

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# instantiate new OpenAI client for Responses API
client = OpenAI()
# fallback legacy model wrapper (used by CodeAgent)
gpt_4_1 = LiteLLMModel(model_id="openai/gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))

# Set page configuration
st.set_page_config(page_title="AortaGPT: Clinical Decision Support Tool", 
                   layout="wide", 
                   page_icon=":anatomical_heart:",
                   initial_sidebar_state="expanded")

# Custom CSS styling
st.markdown("""
    <style>
    .highlight-box {
        background-color: #3b2b2b;
        border-radius: 12px;
        padding: 10px;
        color: white;
        font-weight: 500;
        margin-bottom: 15px;
    }
    .source-link {
        display: inline-block;
        margin-right: 10px;
        padding: 3px 8px;
        background-color: #2b3b4b;
        border-radius: 4px;
        color: white;
        text-decoration: none;
        font-size: 0.9em;
        transition: background-color 0.2s;
    }
    .source-link:hover {
        background-color: #3b4b5b;
    }
    .search-container {
        position: relative;
        margin-bottom: 15px;
    }
    .search-results {
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    .search-result-item {
        padding: 8px;
        cursor: pointer;
        border-bottom: 1px solid #eee;
    }
    .search-result-item:hover {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title(":anatomical_heart: AortaGPT: Clinical Decision Support Tool")

# Clinical history options
clinical_options = [
    "Diagnosis of Aortic Aneurysm and/or Dissection",
    "Family History of Aortic Aneurysm or Dissection",
    "Diagnosis of Aneurysm in Other Arteries",
    "Family History of Aneurysm in Other Arteries",
    "Diagnosis of Hypertrophic Cardiomyopathy",
    "Family History of Hypertrophic Cardiomyopathy",
    "Diagnosis of Dilated Cardiomyopathy",
    "Family History of Dilated Cardiomyopathy",
    "Diagnosis of Long QT Syndrome",
    "Family History of Long QT Syndrome",
    "Diagnosis of Dyslipidemia",
    "Family History of Dyslipidemia",
    "Marfanoid Features Present",
    "Loeys-Dietz Features Present",
    "Ehlers-Danlos Features Present",
    "Currently Pregnant or Considering Pregnancy"
]

# Supported gene options for filtering and selection
GENE_OPTIONS = [
    "FBN1", "TGFBR1", "TGFBR2", "SMAD3", "TGFB2", "TGFB3",
    "ACTA2", "MYH11", "MYLK", "PRKG1", "LOX", "COL3A1",
    "SLC2A10", "Other"
]

# Initialize ALL session state variables in one place
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_variant_info' not in st.session_state:
    st.session_state.selected_variant_info = None
if 'variant_cache' not in st.session_state:
    st.session_state.variant_cache = {}
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'filtered_variants' not in st.session_state:
    st.session_state.filtered_variants = []
if 'other_relevant_details' not in st.session_state:
    st.session_state.other_relevant_details = ""
if 'chat_active' not in st.session_state:
    st.session_state.chat_active = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
# Note: Text interpretation state is now managed by TextInterpretationManager

# Initialize text interpretation manager
interpretation_manager = TextInterpretationManager(client, GENE_OPTIONS, clinical_options)

# Sidebar input panel
with st.sidebar:
    st.header("Patient Input Parameters")

    # Free-text agent input using new architecture
    interpretation_manager.render_ui()

    # Demographics
    st.subheader("Demographics")
    age = st.number_input("Age (years)", 0, 120, st.session_state.get("age", 30), 1, key="age")
    sex = st.selectbox("Sex", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.get("sex", "Male")), key="sex")

    # Genetic Profile
    st.subheader("Genetic Profile")
    # Gene selection (pre-filtered via model); use global GENE_OPTIONS list
    gene = st.selectbox(
        "Gene",
        GENE_OPTIONS,
        index=GENE_OPTIONS.index(st.session_state.get("gene", "FBN1")),
        key="gene"
    )
    if gene == "Other":
        gene = st.text_input("Enter Gene Name", st.session_state.get("gene", ""), key="custom_gene")

    # Variant selection for chosen gene (lazy-loaded in background)
    st.subheader("Variant")
    variant = st.session_state.get("variant", "")
    if gene and gene != "Other":
        cache_key = f"variants_{gene}"
        # Initialize empty list and spawn background fetch on first access
        if cache_key not in st.session_state.variant_cache:
            st.session_state.variant_cache[cache_key] = []
            def _load_variants(g):
                vs = fetch_clinvar_variants(g)
                if 'variant_cache' not in st.session_state:
                    st.session_state.variant_cache = {}
                st.session_state.variant_cache[f"variants_{g}"] = vs
            threading.Thread(target=_load_variants, args=(gene,), daemon=True).start()
        variants_list = st.session_state.variant_cache[cache_key]
        # Search/filter
        search_query = st.text_input("Search variants", value=variant, key="variant_search")
        filtered_variants = filter_variants(variants_list, search_query)
        # Default index if model suggested exists
        default_idx = 0
        if variant in filtered_variants:
            default_idx = filtered_variants.index(variant)
        variant_selection = st.selectbox("Select Variant", filtered_variants, index=default_idx, key="variant_select")
        if variant_selection == "Enter custom variant":
            variant = st.text_input("Enter custom variant", value=variant, key="custom_variant")
            st.session_state.selected_variant_info = None
        else:
            variant = variant_selection
            st.session_state.selected_variant_info = fetch_variant_details(variant)
    else:
        variant = st.text_input("Enter custom variant", value=variant, key="variant")
        st.session_state.selected_variant_info = None
    st.session_state["variant"] = variant

    # Aortic Measurements
    st.subheader("Aortic Measurements")
    # Ensure numeric defaults are floats to avoid mixed-type errors
    root_diameter = st.number_input(
        "Aortic Root Diameter (mm)",
        0.0,
        100.0,
        float(st.session_state.get("root_diameter", 45.0)),
        0.1,
        key="root_diameter"
    )
    ascending_diameter = st.number_input(
        "Ascending Aorta Diameter (mm)",
        0.0,
        100.0,
        float(st.session_state.get("ascending_diameter", 0.0)),
        0.1,
        key="ascending_diameter"
    )
    z_score = st.number_input(
        "Z-score (if available)",
        value=float(st.session_state.get("z_score", 0.0)),
        step=0.1,
        key="z_score"
    )

    # Medications
    st.subheader("Medications")
    meds_options = [
        "Beta-blocker", "ACE-Inhibitor", "ARB", "Statin",
        "Insulin", "Metformin", "Antiplatelet", "Diuretic", "Other"
    ]
    # Normalize any pre-set meds to match available options
    saved_meds = st.session_state.get("meds", []) or []
    canonical_meds = []
    for m in saved_meds:
        nm = m.lower().replace('-', '').replace(' ', '')
        matched = False
        for opt in meds_options:
            if nm == opt.lower().replace('-', '').replace(' ', ''):
                canonical_meds.append(opt)
                matched = True
                break
        if not matched:
            canonical_meds.append("Other")
    # Update session state to safe defaults
    st.session_state["meds"] = canonical_meds
    meds = st.multiselect(
        "Select Current Medications", meds_options,
        default=canonical_meds,
        key="meds"
    )

    # Clinical History
    st.subheader("Clinical History")
    hx = []
    for option in clinical_options:
        if st.checkbox(option, value=st.session_state.get(option, False), key=option):
            hx.append(option)
    # Other relevant details
    st.subheader("Other Relevant Details")
    other_relevant = st.text_area(
        "Other Relevant Details",
        value=st.session_state.get("other_relevant_details", ""),
        key="other_relevant_details"
    )

## Main content tabs

tabs = st.tabs(["📋 Report", "💬 Chat"])

# RAG Pipeline


# Report tab (first tab)
with tabs[0]:
    st.header("📋 Comprehensive Clinical Report")
    
    # Initialize report generator
    report_generator = ReportGenerator(client)
    
    # Initialize report state
    if 'generated_report' not in st.session_state:
        st.session_state.generated_report = None
    if 'report_timestamp' not in st.session_state:
        st.session_state.report_timestamp = None
    
    # Generate Report button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🚀 Generate Comprehensive Report", type="primary", use_container_width=True):
            # Generate the report
            report_data = report_generator.generate_report(st.session_state, clinical_options)
            
            if report_data:
                st.session_state.generated_report = report_data
                st.session_state.report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("✅ Report generated successfully!")
                st.rerun()
    
    # Display existing report if available
    if st.session_state.generated_report:
        with col2:
            # Export button
            patient_info = {
                'age': age,
                'sex': sex,
                'gene': gene,
                'variant': variant,
                'root_diameter': root_diameter,
                'ascending_diameter': ascending_diameter
            }
            export_text = report_generator.export_report(st.session_state.generated_report, patient_info)
            
            st.download_button(
                label="📥 Download Report",
                data=export_text,
                file_name=f"AortaGPT_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.session_state.generated_report = None
                st.session_state.report_timestamp = None
                st.rerun()
        
        # Show generation timestamp
        st.caption(f"Generated: {st.session_state.report_timestamp}")
        st.divider()
        
        # Display the structured report
        report_generator.display_structured_report(st.session_state.generated_report)
        
        # Add disclaimer
        st.warning("""
        **DISCLAIMER:** This report provides informational guidance only and does not replace clinical judgment. 
        All recommendations should be reviewed by qualified healthcare providers familiar with the patient's complete history. 
        The information is based on current guidelines but may not account for all individual factors.
        """)
    else:
        # Show placeholder when no report is generated
        st.info("""
        Click the **Generate Comprehensive Report** button above to create a detailed clinical report based on the patient parameters you've entered.
        
        The report will include:
        - Initial workup recommendations
        - Risk stratification with modifier score
        - Surgical thresholds
        - Imaging surveillance schedule
        - Lifestyle guidelines
        - Pregnancy/peripartum management
        - Genetic counseling recommendations
        - Blood pressure targets
        - Medication management
        - Gene/variant interpretation
        
        All recommendations will be evidence-based and include specific citations to approved clinical sources.
        """)

# Chat tab (second tab)
with tabs[1]:
    st.header("💬 AortaGPT Chat")
    
    # Initialize Chat button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🚀 Initialize Chat", type="primary", use_container_width=True, key="init_chat_button"):
            st.session_state.chat_active = True
            
            # Perform vector search based on current patient context
            with st.spinner("Searching relevant medical literature..."):
                try:
                    # Build patient context from current session state
                    context_str = build_patient_context(st.session_state, clinical_options)
                    
                    # Perform vector search
                    results = search_documents(
                        query=context_str,
                        index_path="data/embeddings.pkl",
                        top_k=5,
                        snippet_length=200
                    )
                    st.session_state.search_results = results
                except Exception as e:
                    st.error(f"Error searching documents: {e}")
                    st.session_state.search_results = []
            
            # Build dynamic context from retrieved documents
            docs = st.session_state.get("search_results", []) or []
            context_lines = []
            for doc in docs:
                context_lines.append(
                    f"Source: {doc.get('file','Unknown')}"
                    f" (score: {doc.get('score',0):.3f})\n"
                    f"Snippet: {doc.get('snippet','')}"
                )
            dynamic_context = "\n\n".join(context_lines)
            
            # Build patient context string for the chat
            patient_context = build_patient_context(st.session_state, clinical_options)
            
            # Combine chat system prompt with patient info and dynamic context
            initial_prompt = (
                chat_system_prompt.strip() + 
                "\n\n## CURRENT PATIENT INFORMATION:\n" + 
                patient_context + 
                "\n\n## RETRIEVED MEDICAL CONTEXT:\n" + 
                dynamic_context
            )
            st.session_state.chat_history = [{"role": "system", "content": initial_prompt}]
            st.rerun()
    
    if st.session_state.chat_active:
        # Render chat history (skip system messages)
        for msg in st.session_state.chat_history:
            if msg["role"] != "system":
                st.chat_message(msg["role"]).write(msg["content"])
        
        user_input = st.chat_input("Ask a question about this patient...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("AortaGPT is thinking..."):
                response = client.responses.create(
                    model="gpt-4.1-nano",
                    input=st.session_state.chat_history
                )
            assistant_msg = ''
            if getattr(response, 'output_text', None):
                assistant_msg = response.output_text
            else:
                for m in getattr(response, 'output', []):
                    for chunk in m.get('content', []):
                        if chunk.get('type') == 'output_text':
                            assistant_msg += chunk.get('text', '')
                        elif chunk.get('type') == 'refusal':
                            assistant_msg += chunk.get('refusal', '')
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
            st.rerun()
    else:
        # Show placeholder when chat is not initialized
        st.info("""
        Click the **Initialize Chat** button above to start a conversation with AortaGPT.
        
        The chat will:
        - Search relevant medical literature based on patient parameters
        - Provide context-aware responses specific to this patient
        - Reference evidence-based guidelines and research
        - Maintain conversation history throughout the session
        
        You can ask questions about diagnosis, treatment options, risk assessment, and clinical management.
        """)