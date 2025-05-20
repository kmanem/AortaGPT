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


# Initialize session state
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
# Initialize session state
if 'selected_variant_info' not in st.session_state:
    st.session_state.selected_variant_info = None
# Initialize variant cache in session state
# Initialize variant cache in session state
if 'variant_cache' not in st.session_state:
    st.session_state.variant_cache = {}
# Initialize other relevant details field
# Initialize other relevant details and chat state
if 'other_relevant_details' not in st.session_state:
    st.session_state.other_relevant_details = ""
if 'chat_active' not in st.session_state:
    st.session_state.chat_active = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []

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



# Function to interpret a free-text description via OpenAI Responses API and apply parameters
def interpret_and_apply_params(description: str) -> None:
    """
    Parse a free-form patient description using the OpenAI Responses API and update session state.
    """
    # Build a simple system/user messages sequence
    system_msg = (
        "You are a helpful assistant that extracts patient parameters from a free-form description. "
        "Also populate 'other_relevant_details' with any additional clinically relevant details."
    )
    user_msg = description
    # Narrow gene choices based on description to reduce options
    candidates = [g for g in GENE_OPTIONS if g != "Other" and g.lower() in description.lower()]
    gene_enum = candidates + ["Other"] if candidates else GENE_OPTIONS.copy()
    # Build JSON schema for structured output
    schema = {
        "type": "object",
        "properties": {
            "age": {"type": "integer"},
            "sex": {"type": "string", "enum": ["Male", "Female", "Other"]},
            "gene": {"type": "string", "enum": gene_enum},
            "custom_gene": {"type": "string"},
            "variant": {"type": "string"},
            "root_diameter": {"type": "number"},
            "ascending_diameter": {"type": "number"},
            "z_score": {"type": "number"},
            "meds": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "age", "sex", "gene", "custom_gene", "variant",
            "root_diameter", "ascending_diameter", "z_score", "meds"
        ],
        "additionalProperties": False
    }
    # add clinical history booleans
    for opt in clinical_options:
        schema["properties"][opt] = {"type": "boolean"}
        schema["required"].append(opt)
    # include free-text field for any other relevant details
    schema["properties"]["other_relevant_details"] = {"type": "string"}
    schema["required"].append("other_relevant_details")
    # Call the Responses API with structured output
    try:
        response = client.responses.create(
            model="gpt-4.1-nano",
            input=[
                {"role": "system",  "content": system_msg},
                {"role": "user",    "content": user_msg}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "patient_config",
                    "schema": schema,
                    "strict": True
                }
            }
        )
        # Extract the JSON output text and parse
        print(response.output_text)
        raw = response.output_text
        config = json.loads(raw)
        # Build a patient summary string for retrieval
        context_str = build_patient_context(config, clinical_options)
        # Perform vector search on summary context
        results = search_documents(
            query=context_str,
            index_path="data/embeddings.pkl",
            top_k=5,
            snippet_length=200
        )
        st.session_state.search_results = results
    except Exception as e:
        st.error(f"Error parsing parameters: {e}")
        return
    # Apply via existing tool (this will rerun)
    configure_all_params(config)

# Sidebar input panel
with st.sidebar:
    st.header("Patient Input Parameters")

    # Free-text agent input
    with st.expander("\U0001F5E3 Set Parameters from Text"):
        user_prompt = st.text_area("Describe the patient")
        if st.button("Interpret & Apply Parameters"):
            with st.spinner("Interpreting parameters..."):
                interpret_and_apply_params(user_prompt)

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
submitted = st.sidebar.button("Initialize Chat", type="primary")
tabs = st.tabs([":speech_balloon: Chat", ":page_facing_up: Report"])

# RAG Pipeline


with tabs[0]:
    # Chat setup
    if submitted:
        st.session_state.chat_active = True
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
        # Combine static system prompt with dynamic context
        initial_prompt = system_prompt.strip() + "\n\n" + dynamic_context
        st.session_state.chat_history = [{"role": "system", "content": initial_prompt}]
    if st.session_state.chat_active:
        st.header(":speech_balloon: AortaGPT Chat")
        # Display retrieved context documents
        with st.expander("📚 Retrieved Context Documents", expanded=False):
            query_ctx = st.text_input("Search within context", key="context_search")
            docs = st.session_state.get("search_results", []) or []
            # Filter by search query in snippet or filename
            if query_ctx:
                docs = [d for d in docs if query_ctx.lower() in d.get('snippet','').lower() or query_ctx.lower() in d.get('file','').lower()]
            for d in docs:
                st.markdown(f"**{d.get('file','Unknown')}** (score: {d.get('score',0):.3f})")
                st.write(d.get('snippet',''))
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

with tabs[1]:
    # Save to history
    current_session = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'gene': gene,
        'variant': variant,
        'root_diameter': root_diameter,
        'age': age,
        'sex': sex
    }
    st.session_state.history.append(current_session)

    # Display sections
    st.subheader(":chart_with_upwards_trend: Risk Stratification")
    display_risk_stratification(gene, variant, root_diameter, z_score, hx)

    st.subheader("🏥 Surgical Thresholds")
    display_surgical_thresholds(gene, root_diameter)

    st.subheader("🩻 Imaging Surveillance")
    display_imaging_surveillance(gene, root_diameter, ascending_diameter, hx)

    st.subheader("🏃 Lifestyle & Activity Guidelines")
    display_lifestyle_guidelines(gene, sex, hx)

    st.subheader(":family: Genetic Counseling")
    display_genetic_counseling(gene, variant, sex, hx)

    st.subheader(":rotating_light: Red Flag Alerts")
    display_red_flag_alerts(gene, root_diameter, ascending_diameter, hx)

    st.subheader(":bar_chart: Kaplan-Meier Survival Curve")
    fig = display_kaplan_meier(gene, age, sex)
    st.pyplot(fig)

    # Add references with clickable links
    with st.expander("References & Sources"):
        st.markdown("""
        **Guidelines and Resources:**
        <ul>
            <li><a href="https://www.ahajournals.org/doi/10.1161/CIR.0000000000001106" target="_blank">2022 ACC/AHA Aortic Disease Guidelines</a></li>
            <li><a href="https://www.clinicalgenome.org/" target="_blank">ClinGen</a></li>
            <li><a href="https://www.ncbi.nlm.nih.gov/books/NBK1116/" target="_blank">GeneReviews</a></li>
            <li><a href="https://www.marfan.org/" target="_blank">The Marfan Foundation</a></li>
            <li><a href="https://ehlers-danlos.com/" target="_blank">The Ehlers-Danlos Society</a></li>
        </ul>
        
        **Risk Calculations:**
        <p>Risk calculations are based on published literature for specific genes and follow current consensus guidelines.</p>
        """, unsafe_allow_html=True)

    # Add disclaimer
    st.warning("""
    **DISCLAIMER:** This tool provides informational guidance only and does not replace clinical judgment. 
    All recommendations should be reviewed by qualified healthcare providers familiar with the patient's complete history. 
    The information is based on current guidelines but may not account for all individual factors.
    """)

    # Show history
    with st.expander("Previous Sessions"):
        if st.session_state.history:
            for i, session in enumerate(st.session_state.history):
                st.write(f"**Session {i+1}:** {session['timestamp']}")
                st.write(f"Gene: {session['gene']}, Variant: {session['variant']}, Root: {session['root_diameter']}mm")
                st.divider()
        else:
            st.write("No previous sessions")