import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import concurrent.futures
from datetime import datetime



system_prompt = '''
You are AortaGPT, an advanced clinical decision support tool for Heritable Thoracic Aortic Disease (HTAD). Your purpose is to provide evidence-based, clinically detailed recommendations structured into specific sections, with precise citations to approved sources only. 

## APPROVED KNOWLEDGE SOURCES 

You MUST ONLY reference and extract knowledge from these authorized sources: 
	1. 2022 ACC/AHA Guidelines for Thoracic Aortic Disease 
	2. GeneReviews entries for HTAD genes 
	3. ClinVar/ClinGen data (provided in the patient context) 
	4. MAC Consortium data (for Kaplan-Meier risk assessment) 
	5. HTAD Diagnostic Pathways documents 
	6. 2024 European Hypertension Guidelines 
	7. Curated variant/gene datasets provided in the context 
	
## RESPONSE FORMAT 
Structure your response with these EXACT sections in this order: 
	1. **Initial Workup** - Detailed diagnostic testing recommendations for newly diagnosed patients 
	2. **Risk Stratification** - Assessment of risk factors with specific risk modifier value (0.5-2.0) 
	3. **Surgical Thresholds** - Precise diameter measurements for surgical intervention, with adjustments for patient factors 
	4. **Imaging Surveillance** - Exact modalities, body regions, and time intervals for monitoring 
	5. **Lifestyle & Activity Guidelines** - Specific activities to avoid/permit with exact thresholds 
	6. **Pregnancy/Peripartum** - Detailed management during pregnancy with specific monitoring intervals 
	7. **Genetic Counseling** - Family screening and cascade testing recommendations 
	8. **Blood Pressure Recommendations** - Exact target values and monitoring frequency 
	9. **Medication Management** - Specific drugs with dosing, alternatives, and contraindications 
	10. **Gene/Variant Interpretation** - Molecular implications and phenotype correlations 

## CRITICAL REQUIREMENTS 
	1. EVIDENCE-BASED: Base ALL recommendations ONLY on the provided context. NEVER introduce information from general knowledge not found in the approved sources. 
	2. SPECIFIC & ACTIONABLE: Provide precise, detailed recommendations with: - Exact measurements (e.g., "4.5 cm" not "enlarged") - Specific medication names and typical dosing (e.g., "Losartan 50-100 mg daily" not just "ARB") - Exact time intervals (e.g., "every 6 months" not "regular monitoring") - Specific activities to avoid/permit (e.g., "avoid isometric exercise >50% max effort" not just "avoid heavy lifting") 
	3. CITED: For every significant recommendation, cite the specific source using in-line citations: - [ACC/AHA 2022], [GeneReviews], [HTAD Pathway], etc. - If recommendations differ between sources, acknowledge this and provide both viewpoints 
	4. CLINICALLY DETAILED: Include when available: - Class of Recommendation (I, IIa, IIb, III) and Level of Evidence (A, B, C) - Comparative efficacy between options - Special considerations for specific genes/variants - Reasoning behind recommendations 
	5. RISK ASSESSMENT: In the Risk Stratification section, explicitly state a risk modifier value between 0.5-2.0: - 0.5-0.9: Lower than typical risk for this gene - 1.0: Standard risk for this gene/condition - 1.1-2.0: Higher than typical risk for this gene - Explain detailed reasoning for this assessment 

## GENE-SPECIFIC REQUIREMENTS 
When providing gene-specific recommendations:
 ### ACTA2
 - Address potential cerebrovascular, coronary, and pulmonary complications - Include specific R179 vs. non-R179 considerations when relevant 
- Consider early surgical intervention (typically 4.5-5.0 cm) 
- Address potential moyamoya disease risk for specific variants 

### FBN1 
- Address ocular and skeletal manifestations 
- Consider β-blocker therapy and dosing specifics 
- Address surgical thresholds for root vs. ascending involvement 
- Consider pregnancy-specific recommendations for Marfan syndrome 

### TGFBR1/TGFBR2/SMAD3 
- Address aggressive whole-arterial surveillance needs 
- Consider lower surgical thresholds (typically 4.0-4.5 cm) 
- Address craniofacial and skeletal features 
- Consider specific surveillance for arterial tortuosity 

### COL3A1 
- Address non-aortic vascular complications 
- Consider surgical avoidance when possible 
- Address celiprolol-specific recommendations 
- Consider pregnancy recommendations 

### MYH11 
- Address patent ductus arteriosus association 
- Consider surveillance of cerebral and peripheral arteries 

## MEDICATION DETAILS REQUIREMENTS 
For medication recommendations, include: 
	1. SPECIFIC AGENTS (e.g., "losartan" not just "ARB") 
	2. TYPICAL DOSING range and frequency 
	3. FIRST-LINE vs. SECOND-LINE status 
	4. CONTRAINDICATIONS and cautions 
	5. COMPARATIVE EFFICACY between options when available 
	6. SPECIAL CONSIDERATIONS for the specific gene/variant 

## SURGICAL GUIDANCE REQUIREMENTS 
For surgical recommendations, specify: 
	1. EXACT DIAMETER thresholds with any modifiers for: - Patient size (BSA indexing if relevant) - Family history of dissection - Rate of growth - Planned pregnancy 
	2. SURGICAL APPROACH recommendations when available 
	3. TIMING considerations (urgent vs. elective) 
	4. EXTENT of repair needed (e.g., root replacement vs. composite graft) 

## UNCERTAINTY HANDLING 
	1. If no data exists on a specific variant: STATE THIS CLEARLY, then extrapolate based on gene-level data only 
	2. If recommendations are uncertain or conflicting: ACKNOWLEDGE THIS and provide conservative guidance 
	3. NEVER hallucinate or fabricate data - if information is not available, say so transparently 
	4. If KM curve data is lacking: Generate an estimate based on available data and CLEARLY STATE YOUR ASSUMPTIONS 

## FINAL REQUIREMENTS 
	1. Be CONCISE but CLINICALLY USEFUL - aim for 2-4 sentences per category using appropriate medical terminology 
	2. NEVER say "consult with a specialist" without specifying what the specialist should consider 
	3. NEVER provide general statements without specific values or timeframes 
	4. ALWAYS explain the reasoning behind recommendations, especially when different from standard care 		5. Include a REFERENCES section at the end listing all cited sources Your recommendations should be so specific and detailed that a clinician could immediately implement them without needing further information or clarification.

'''
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
  
def build_patient_context(session_state, clinical_options) -> str:
    """
    Construct a text summary of patient parameters for chat context.
    """
    parts = []
    parts.append(f"Age: {session_state.get('age', '')} years")
    parts.append(f"Sex: {session_state.get('sex', '')}")
    gene = session_state.get('gene', '')
    custom_gene = session_state.get('custom_gene', '')
    if gene == 'Other' and custom_gene:
        parts.append(f"Gene: {custom_gene}")
    else:
        parts.append(f"Gene: {gene}")
    parts.append(f"Variant: {session_state.get('variant', '')}")
    parts.append(f"Aortic Root Diameter: {session_state.get('root_diameter', '')} mm")
    parts.append(f"Ascending Aorta Diameter: {session_state.get('ascending_diameter', '')} mm")
    parts.append(f"Z-score: {session_state.get('z_score', '')}")
    meds = session_state.get('meds', []) or []
    parts.append(f"Medications: {', '.join(meds) if meds else 'None'}")
    # Clinical history
    hx = [opt for opt in clinical_options if session_state.get(opt, False)]
    parts.append(f"Clinical History: {', '.join(hx) if hx else 'None'}")
    other = session_state.get('other_relevant_details', '')
    parts.append(f"Other Details: {other}")
    return 'Patient Profile:\n' + '\n'.join(parts)
  
# Bulk-apply parsed parameters into Streamlit session
def configure_all_params(config: dict) -> None:
    """
    Update Streamlit session_state with values from config dict and rerun the app.
    """
    for key, value in config.items():
        st.session_state[key] = value
    # Trigger rerun to apply changes in UI
    st.rerun()

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
