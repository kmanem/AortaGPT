import streamlit as st
import openai
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from smolagents import LiteLLMModel  # smolagents imports retained for future use
import threading
from helper_functions import *

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# instantiate new OpenAI client for Responses API
client = OpenAI()
# fallback legacy model wrapper (used by CodeAgent)
gpt_4_1 = LiteLLMModel(model_id="openai/gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))

# Set page configuration
st.set_page_config(page_title="AortaGPT: Clinical Decision Support Tool", layout="wide", page_icon=":anatomical_heart:")

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

# Display variant information
if st.session_state.selected_variant_info:
    variant_info = st.session_state.selected_variant_info
    with st.expander("Variant Information"):
        st.markdown(f"""
        **Selected Variant:** {variant}  
        **Clinical Significance:** {variant_info.get('clinical_significance', 'Not available')}  
        **Review Status:** {variant_info.get('review_status', 'Not available')}  
        **Last Updated:** {variant_info.get('last_updated', 'Not available')}  
        <a href="{variant_info.get('clinvar_url', '#')}" target="_blank">View in ClinVar</a>
        """, unsafe_allow_html=True)

# Initialize Chat button
submitted = st.sidebar.button("Initialize Chat", type="primary")
if submitted:
    # Activate chat interface and seed context
    st.session_state.chat_active = True
    from helper_functions import build_patient_context
    context = build_patient_context(st.session_state, clinical_options)
    st.session_state.chat_history = [{"role": "system", "content": context}]

# Chat interface activated?
if st.session_state.chat_active:
    st.header(":speech_balloon: AortaGPT Chat")
    # Display past messages
    for msg in st.session_state.chat_history:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).write(msg["content"])
    # Get user input
    user_input = st.chat_input("Ask a question about this patient...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        # Send through Responses API for structured chat
        with st.spinner("AortaGPT is thinking..."):
            response = client.responses.create(
                model="gpt-4.1",
                input=st.session_state.chat_history
            )
        # Parse assistant message
        if hasattr(response, 'output_text') and response.output_text:
            assistant_msg = response.output_text
        else:
            assistant_msg = ''
            try:
                for msg in response.output:
                    for chunk in msg['content']:
                        if chunk.get('type') == 'output_text':
                            assistant_msg += chunk.get('text', '')
                        elif chunk.get('type') == 'refusal':
                            assistant_msg += chunk.get('refusal', '')
            except Exception:
                assistant_msg = '<Could not parse model response>'
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
        # Re-run to display the assistant's message
        st.rerun()

