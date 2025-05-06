import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import concurrent.futures
from datetime import datetime
from functions import *

# Set page configuration
st.set_page_config(page_title="AortaGPT: Clinical Decision Support Tool", layout="wide", page_icon=":anatomical_heart:")

# Apply custom styling
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
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'filtered_variants' not in st.session_state:
    st.session_state.filtered_variants = []

# Main App Layout
# Sidebar for patient inputs
with st.sidebar:
    st.header("Patient Input Parameters")
    
    # Demographics
    st.subheader("Demographics")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    
    # Genetic Profile
    st.subheader("Genetic Profile")
    gene = st.selectbox("Gene", ["FBN1", "TGFBR1", "TGFBR2", "SMAD3", "TGFB2", "TGFB3", "ACTA2", "MYH11", "MYLK", "PRKG1", "LOX", "COL3A1", "SLC2A10", "Other"])
    if gene == "Other":
        gene = st.text_input("Enter Gene Name")

    # Improved variant selection with search
    variant = None
    if gene and gene != "Other":
        # Show progress indicator
        with st.spinner("Loading variants..."):
            variants_list = fetch_clinvar_variants(gene)
        
        if variants_list:
            # Create search box
            st.write("### Search for Variant")
            search_query = st.text_input("Type to search variants", key="variant_search")
            
            # Filter variants based on search
            filtered_variants = filter_variants(variants_list, search_query)
            
            # Display filtered results
            if filtered_variants:
                variant_selection = st.selectbox(
                    "Select Variant", 
                    filtered_variants,
                    index=0
                )
                
                # Handle custom variant
                if variant_selection == "Enter custom variant":
                    variant = st.text_input("Enter custom variant")
                else:
                    variant = variant_selection
                    
                    # Fetch variant details
                    with st.spinner("Loading variant details..."):
                        st.session_state.selected_variant_info = fetch_variant_details(variant)
            else:
                st.info("No variants found matching your search. Try different keywords.")
        else:
            st.info(f"No variants found in ClinVar for {gene}. Please enter variant manually.")
            variant = st.text_input("Enter Variant (No ClinVar variants found)")
    else:
        variant = st.text_input("Enter Variant")
        
    # Aortic Measurements
    st.subheader("Aortic Measurements")
    root_diameter = st.number_input("Aortic Root Diameter (mm)", min_value=0.0, max_value=100.0, value=45.0, step=0.1)
    ascending_diameter = st.number_input("Ascending Aorta Diameter (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    z_score = st.number_input("Z-score (if available)", value=0.0, step=0.1)

    # Medications Section
    st.subheader("Medications")
    meds = st.multiselect("Select Current Medications", [
        "Beta-blocker", "ACE-Inhibitor", "ARB", "Statin", "Insulin", "Metformin", "Antiplatelet", "Diuretic", "Other"])

    # Clinical History - Restored all options
    st.subheader("Clinical History")
    hx = []
    options = [
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
    for option in options:
        if st.checkbox(option):
            hx.append(option)

# Display variant information in main area
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

# Generate button
submitted = st.sidebar.button("Generate AortaGPT Output", type="primary")

# Validate measurements
measurement_errors = validate_measurements(root_diameter, ascending_diameter)
if measurement_errors:
    for error in measurement_errors:
        st.error(error)

# Generate output when submitted
if submitted and not measurement_errors:
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