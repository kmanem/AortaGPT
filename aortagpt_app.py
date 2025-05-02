import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import concurrent.futures
from datetime import datetime

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
if 'variant_cache' not in st.session_state:
    st.session_state.variant_cache = {}
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'filtered_variants' not in st.session_state:
    st.session_state.filtered_variants = []

# Improved API functions
def rate_limited_api_call(url, params, max_retries=3):
    """Make API calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"API call failed: {str(e)}")
                return None
            time.sleep(1)
    
    return None

@st.cache_data(ttl=3600)
def fetch_clinvar_variants(gene_symbol):
    """Fetch variants for a given gene from ClinVar API with parallel processing"""
    try:
        # Check cache first
        cache_key = f"variants_{gene_symbol}"
        if cache_key in st.session_state.variant_cache:
            return st.session_state.variant_cache[cache_key]
        
        # NCBI E-utilities API endpoint
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        # Build query
        query = f"{gene_symbol}[gene] AND (\"pathogenic\"[clinical_significance] OR \"likely pathogenic\"[clinical_significance])"
        
        # Request parameters
        params = {
            "db": "clinvar",
            "term": query,
            "retmode": "json",
            "retmax": 500
        }
        
        # Make the request
        response = rate_limited_api_call(base_url, params)
        if not response:
            return []
            
        # Get list of variant IDs
        id_list = response.get('esearchresult', {}).get('idlist', [])
        
        if not id_list:
            return []
            
        # Fetch details for variants in parallel
        variants = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for variant_id in id_list:
                futures.append(
                    executor.submit(fetch_variant_name, variant_id)
                )
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    variants.append(result)
        
        # Add custom option
        variants.append("Enter custom variant")
        
        # Cache results
        st.session_state.variant_cache[cache_key] = variants
        return variants
        
    except Exception as e:
        st.warning(f"Error fetching variants: {str(e)}")
        return []

def fetch_variant_name(variant_id):
    """Helper function to fetch a single variant name"""
    try:
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {
            "db": "clinvar",
            "id": variant_id,
            "retmode": "json"
        }
        
        summary_response = rate_limited_api_call(summary_url, summary_params)
        if not summary_response:
            return None
            
        result = summary_response.get('result', {})
        if variant_id in result:
            variant_info = result[variant_id]
            variant_name = variant_info.get('title', '')
            if variant_name:
                return variant_name
        
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_variant_details(variant_name):
    """Fetch detailed information about a specific variant"""
    try:
        # Check cache
        cache_key = f"details_{variant_name}"
        if cache_key in st.session_state.variant_cache:
            return st.session_state.variant_cache[cache_key]
            
        # Search for variant
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "clinvar",
            "term": variant_name,
            "retmode": "json",
            "retmax": 1
        }
        
        response = rate_limited_api_call(base_url, params)
        if not response:
            return None
            
        id_list = response.get('esearchresult', {}).get('idlist', [])
        
        if not id_list:
            return None
            
        # Get details
        variant_id = id_list[0]
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {
            "db": "clinvar",
            "id": variant_id,
            "retmode": "json"
        }
        
        summary_response = rate_limited_api_call(summary_url, summary_params)
        if not summary_response:
            return None
            
        # Extract information
        result = summary_response.get('result', {})
        if variant_id in result:
            variant_info = result[variant_id]
            
            details = {
                "clinical_significance": variant_info.get('clinical_significance', 'Not available'),
                "review_status": variant_info.get('review_status', 'Not available'),
                "last_updated": variant_info.get('update_date', 'Not available'),
                "variant_id": variant_id,
                "sources": [],
                "clinvar_url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variant_id}/"
            }
            
            # Cache results
            st.session_state.variant_cache[cache_key] = details
            return details
        
        return None
        
    except Exception as e:
        st.warning(f"Error fetching variant details: {str(e)}")
        return None

def filter_variants(variants, query):
    """Filter variants based on search query"""
    if not query:
        return variants
    
    query = query.lower()
    filtered = [v for v in variants if query in v.lower()]
    
    # Always include custom option
    if "Enter custom variant" not in filtered:
        filtered.append("Enter custom variant")
        
    return filtered

# Validation and calculation functions
def validate_measurements(root_diameter, ascending_diameter):
    """Validate aortic measurements"""
    errors = []
    
    if root_diameter <= 0:
        errors.append("Aortic root diameter must be greater than 0")
        
    if ascending_diameter < 0:
        errors.append("Ascending aorta diameter cannot be negative")
        
    if root_diameter > 100 or ascending_diameter > 100:
        errors.append("Aortic measurements appear unusually large. Please verify.")
        
    return errors

def calculate_gene_risk(gene, age, sex, root_diameter):
    """Calculate risk based on gene and patient factors"""
    # Gene-specific risk modifiers
    gene_risk_map = {
        "FBN1": 1.5,
        "TGFBR1": 2.0,
        "TGFBR2": 2.2,
        "ACTA2": 1.8,
        "MYH11": 1.6,
        "COL3A1": 2.5,
    }
    
    # Default for genes not in our map
    gene_modifier = gene_risk_map.get(gene, 1.0)
    
    # Size risk
    size_risk = 0.0
    if root_diameter > 45:
        size_risk = (root_diameter - 45) * 0.1
    
    # Age risk
    age_risk = 0.01 * age
    
    # Sex risk
    sex_modifier = 1.2 if sex == "Male" else 1.0
    
    # Combine factors
    total_risk = (0.05 + size_risk + age_risk) * gene_modifier * sex_modifier
    
    return min(total_risk, 1.0)  # Cap at 100%

# Output display functions
def display_risk_stratification(gene, variant, root_diameter, z_score, hx):
    """Display risk stratification with clickable sources"""
    # Determine risk level
    risk_level = "high"  # Default
    
    if gene in ["FBN1", "TGFBR1", "TGFBR2", "SMAD3"]:
        if root_diameter >= 45:
            risk_level = "high"
        elif root_diameter >= 40:
            risk_level = "moderate"
        else:
            risk_level = "low"
    elif gene in ["ACTA2", "MYH11"]:
        if root_diameter >= 50:
            risk_level = "high"
        elif root_diameter >= 45:
            risk_level = "moderate"
        else:
            risk_level = "low"
    else:
        if root_diameter >= 50:
            risk_level = "high"
        elif root_diameter >= 45:
            risk_level = "moderate"
        else:
            risk_level = "low"
    
    # Override based on history
    if "Diagnosis of Aortic Aneurysm and/or Dissection" in hx:
        risk_level = "high"
    
    # Format output
    if risk_level == "high":
        color = "#3b2b2b"  # Dark red
        risk_text = "high risk"
    elif risk_level == "moderate":
        color = "#3b3b2b"  # Dark yellow
        risk_text = "moderate risk"
    else:
        color = "#2b3b2b"  # Dark green
        risk_text = "lower risk"
        
    st.markdown(f"""
    <div class='highlight-box' style='background-color: {color};'>
    ⚠️ This patient is categorized as <strong>{risk_text}</strong> based on the gene variant {gene} 
    with a root diameter of {root_diameter} mm.
    <br><br>
    <a href="https://www.ahajournals.org/doi/10.1161/CIR.0000000000001106" target="_blank" class="source-link">2022 ACC/AHA Guidelines</a>
    </div>
    """, unsafe_allow_html=True)

def display_surgical_thresholds(gene, root_diameter):
    """Display surgical thresholds with clickable sources"""
    # Logic for different genes
    threshold = 45  # Default
    
    if gene == "FBN1":  # Marfan
        threshold = 45
        secondary_source = "https://www.marfan.org/resource/expert-advice"
        secondary_name = "Marfan Foundation"
    elif gene in ["TGFBR1", "TGFBR2", "SMAD3"]:  # Loeys-Dietz
        threshold = 42
        secondary_source = "https://www.omim.org/entry/609192"
        secondary_name = "OMIM"
    elif gene == "COL3A1":  # vEDS
        threshold = 40
        secondary_source = "https://ehlers-danlos.com/veds-resources"
        secondary_name = "EDS Society"
    elif gene in ["ACTA2", "MYH11"]:  # Familial TAAD
        threshold = 50
        secondary_source = "https://www.ncbi.nlm.nih.gov/books/NBK1120/"
        secondary_name = "GeneReviews"
    else:
        secondary_source = "https://www.ncbi.nlm.nih.gov/books/NBK1120/"
        secondary_name = "GeneReviews"
    
    st.markdown(f"""
    Surgical consideration is warranted if aortic root > {threshold} mm in {gene}-related HTAD.
    <a href="https://www.ahajournals.org/doi/10.1161/CIR.0000000000001106" target="_blank" class="source-link">ACC/AHA 2022</a>
    <a href="{secondary_source}" target="_blank" class="source-link">{secondary_name}</a>
    """, unsafe_allow_html=True)
    
    # Add specific recommendations
    if root_diameter >= threshold:
        st.markdown(f"""
        <div class='highlight-box'>
        ⚠️ Patient's current measurement ({root_diameter} mm) meets surgical threshold.
        </div>
        """, unsafe_allow_html=True)

def display_imaging_surveillance(gene, root_diameter, ascending_diameter, hx):
    """Generate and display imaging surveillance recommendations with clickable sources"""
    # Default recommendation
    frequency = "annual"
    modality = "MRI/CTA"
    
    # Adjust based on risk factors
    if root_diameter >= 45 or "Diagnosis of Aortic Aneurysm and/or Dissection" in hx:
        frequency = "6-month"
    
    # Special cases
    if "Ehlers-Danlos Features Present" in hx or gene == "COL3A1":
        modality = "MRI (avoid CTA when possible)"
    
    st.markdown(f"""
    {frequency.capitalize()} {modality} recommended, or sooner if growth > 3 mm/year.
    <a href="https://www.ahajournals.org/doi/10.1161/CIR.0000000000001106" target="_blank" class="source-link">ACC/AHA 2022</a>
    <a href="https://www.ncbi.nlm.nih.gov/books/NBK1116/" target="_blank" class="source-link">GeneReviews</a>
    """, unsafe_allow_html=True)

def display_lifestyle_guidelines(gene, sex, hx):
    """Generate and display lifestyle guidelines with clickable sources"""
    # Base recommendations
    guidelines = [
        "Avoid isometric exercise and contact sports.",
        "Maintain blood pressure control.",
        "No stimulants."
    ]
    
    # Gene-specific recommendations
    if gene == "FBN1" or "Marfanoid Features Present" in hx:
        guidelines.append("Avoid activities with rapid changes in atmospheric pressure.")
    
    # Pregnancy considerations
    if sex == "Female" and "Currently Pregnant or Considering Pregnancy" in hx:
        guidelines.append("High-risk OB consultation required during pregnancy.")
    elif sex == "Female":
        guidelines.append("Genetic counseling recommended before pregnancy.")
    
    # Display guidelines
    for guideline in guidelines:
        st.write(f"• {guideline}")
    
    st.markdown("""
    <a href="https://www.ncbi.nlm.nih.gov/books/NBK1116/" target="_blank" class="source-link">GeneReviews</a>
    <a href="https://ehlers-danlos.com/resources" target="_blank" class="source-link">EDS Society</a>
    <a href="https://www.ahajournals.org/doi/10.1161/CIR.0000000000001106" target="_blank" class="source-link">ACC/AHA 2022</a>
    """, unsafe_allow_html=True)

def display_genetic_counseling(gene, variant, sex, hx):
    """Generate and display genetic counseling information with clickable sources"""
    # Inheritance pattern
    inheritance = "autosomal dominant"
    if gene in ["MFAP5", "FBN2"]:
        inheritance = "variable penetrance, autosomal dominant"
    
    st.write(f"• Inheritance pattern: {inheritance}")
    st.write("• Recommend cascade genetic testing for first-degree relatives.")
    st.write("• Clinical screening recommended for family members.")
    
    if variant and variant != "Enter custom variant":
        st.write(f"• Specific variant ({variant}) screening recommended for family.")
    
    # Pregnancy considerations
    if sex == "Female":
        if "Currently Pregnant or Considering Pregnancy" in hx:
            st.markdown("""
            <div class='highlight-box'>
            ⚠️ <strong>Pregnancy Alert:</strong> Patient is pregnant or considering pregnancy. High-risk obstetrical care is strongly recommended.
            </div>
            """, unsafe_allow_html=True)
            st.write("• Frequent cardiovascular monitoring during pregnancy required.")
            st.write("• Consider beta-blocker therapy during pregnancy (consult cardiologist).")
            st.write("• Delivery planning should include cardiovascular specialists.")
        else:
            st.write("• Pre-pregnancy counseling recommended before conception.")
            st.write("• Discuss reproductive options including prenatal testing.")
    
    st.markdown("""
    <a href="https://clinicalgenome.org/" target="_blank" class="source-link">ClinGen</a>
    <a href="https://www.ncbi.nlm.nih.gov/books/NBK1116/" target="_blank" class="source-link">GeneReviews</a>
    """, unsafe_allow_html=True)

def display_red_flag_alerts(gene, root_diameter, ascending_diameter, hx):
    """Generate and display red flag alerts with improved visibility"""
    has_red_flags = False
    red_flags = []
    
    # Check measurements against thresholds
    if gene == "FBN1" and root_diameter >= 50:
        red_flags.append("Root diameter ≥ 50mm indicates high dissection risk.")
        has_red_flags = True
    elif gene in ["TGFBR1", "TGFBR2"] and root_diameter >= 45:
        red_flags.append("Root diameter ≥ 45mm with LDS indicates high dissection risk.")
        has_red_flags = True
    elif gene == "COL3A1" and root_diameter >= 40:
        red_flags.append("Root diameter ≥ 40mm with vEDS indicates very high risk.")
        has_red_flags = True
    
    # Check clinical history
    if "Diagnosis of Aortic Aneurysm and/or Dissection" in hx:
        red_flags.append("Previous aortic events indicate high recurrence risk.")
        has_red_flags = True
    
    # Check for high-risk cardiac conditions
    if "Diagnosis of Hypertrophic Cardiomyopathy" in hx:
        red_flags.append("HCM may increase risk of adverse cardiovascular events.")
        has_red_flags = True
    
    if "Diagnosis of Dilated Cardiomyopathy" in hx:
        red_flags.append("DCM in combination with aortopathy requires close monitoring.")
        has_red_flags = True
        
    if "Diagnosis of Long QT Syndrome" in hx:
        red_flags.append("LQTS may complicate management of beta-blockers for aortopathy.")
        has_red_flags = True
    
    # Pregnancy consideration
    if "Currently Pregnant or Considering Pregnancy" in hx:
        red_flags.append("Pregnancy significantly increases dissection risk.")
        has_red_flags = True
    
    # Display red flags or success message
    if has_red_flags:
        st.markdown("""
        <div class='highlight-box'>
        ⚠️ URGENT ALERT: The following red flags were identified:
        </div>
        """, unsafe_allow_html=True)
        
        for flag in red_flags:
            st.write(f"• {flag}")
        
        st.markdown("""
        <a href="https://www.ahajournals.org/doi/10.1161/CIR.0000000000001106" target="_blank" class="source-link">ACC/AHA 2022</a>
        """, unsafe_allow_html=True)
    else:
        st.success("No immediate red flags detected based on current input.")

def display_kaplan_meier(gene, age, sex):
    """Generate Kaplan-Meier curve"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 80, 1000)
    
    # Gene-specific curves
    if gene == "FBN1":
        baseline = 0.00015
        accel_age = 40
        y = np.where(x < accel_age, 
                    1 - np.exp(-baseline * x**1.5),
                    1 - np.exp(-baseline * accel_age**1.5) * np.exp(-baseline * 3 * (x - accel_age)**1.2))
        title = "Marfan Syndrome (FBN1)"
    elif gene == "TGFBR1" or gene == "TGFBR2":
        baseline = 0.0003
        y = 1 - np.exp(-baseline * x**1.8)
        title = "Loeys-Dietz Syndrome"
    elif gene == "ACTA2":
        baseline = 0.0002
        y = 1 - np.exp(-baseline * x**1.6)
        title = "Familial TAAD (ACTA2)"
    elif gene == "COL3A1":
        baseline = 0.0004
        y = 1 - np.exp(-baseline * x**1.9)
        title = "Vascular Ehlers-Danlos (COL3A1)"
    else:
        baseline = 0.0001
        y = 1 - np.exp(-baseline * x**1.5)
        title = f"Aortic Event Risk: {gene}"
    
    # Plot curve
    ax.plot(x, y, color='#1f77b4', linewidth=3)
    
    # Add reference line
    gen_pop = 1 - np.exp(-0.00005 * x**1.3)
    ax.plot(x, gen_pop, '--', color='gray', alpha=0.7, linewidth=2, label="General Population")
    
    # Add current age marker
    if age > 0:
        # Calculate risk at current age
        if gene == "FBN1":
            if age < accel_age:
                current_risk = 1 - np.exp(-baseline * age**1.5)
            else:
                current_risk = 1 - np.exp(-baseline * accel_age**1.5) * np.exp(-baseline * 3 * (age - accel_age)**1.2)
        elif gene == "TGFBR1" or gene == "TGFBR2":
            current_risk = 1 - np.exp(-baseline * age**1.8)
        elif gene == "ACTA2":
            current_risk = 1 - np.exp(-baseline * age**1.6)
        elif gene == "COL3A1":
            current_risk = 1 - np.exp(-baseline * age**1.9)
        else:
            current_risk = 1 - np.exp(-baseline * age**1.5)
            
        ax.plot(age, current_risk, 'ro', markersize=10)
        ax.annotate(f'Current Age: {age}', 
                   xy=(age, current_risk),
                   xytext=(age+5, current_risk),
                   arrowprops=dict(facecolor='red', shrink=0.05, width=2),
                   fontsize=12, fontweight='bold')
    
    # Styling
    ax.set_xlabel("Age (years)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cumulative Risk of Aortic Event", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    # Add confidence interval
    ci_upper = np.minimum(y * 1.2, 1.0)
    ci_lower = y * 0.8
    ax.fill_between(x, ci_lower, ci_upper, color='#1f77b4', alpha=0.2)
    
    return fig

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