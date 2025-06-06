"""
Report generation module for AortaGPT.
Generates comprehensive clinical reports based on patient parameters.
"""
import streamlit as st
from typing import Dict, Any, List
from openai import OpenAI
from helper_functions import build_patient_context
from vector_search import search_documents
from km_curve_generator import KMCurveGenerator
import json
from datetime import datetime


REPORT_SYSTEM_PROMPT = """You are AortaGPT, an advanced clinical decision support tool for Heritable Thoracic Aortic Disease (HTAD). Your purpose is to provide evidence-based, clinically detailed recommendations structured into specific sections, with precise citations to approved sources only. 

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
    4. ALWAYS explain the reasoning behind recommendations, especially when different from standard care
    5. Include a REFERENCES section at the end listing all cited sources

Your recommendations should be so specific and detailed that a clinician could immediately implement them without needing further information or clarification."""


class ReportGenerator:
    """Generates comprehensive clinical reports for HTAD patients."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.km_generator = KMCurveGenerator(client)
        
    def generate_report(self, session_state: Dict[str, Any], clinical_options: List[str]) -> Dict[str, str]:
        """
        Generate a comprehensive clinical report based on patient parameters.
        
        Args:
            session_state: Streamlit session state containing patient data
            clinical_options: List of clinical history options
            
        Returns:
            Generated report as structured dictionary
        """
        # Build patient context
        patient_context = build_patient_context(session_state, clinical_options)
        
        # Perform vector search for relevant medical literature
        with st.spinner("Searching medical literature for report generation..."):
            try:
                results = search_documents(
                    query=patient_context,
                    index_path="data/embeddings.pkl",
                    top_k=10,  # Get more results for comprehensive report
                    snippet_length=300
                )
            except Exception as e:
                st.error(f"Error searching documents: {e}")
                results = []
        
        # Build retrieved context
        context_lines = []
        for doc in results:
            context_lines.append(
                f"Source: {doc.get('file','Unknown')}\n"
                f"Content: {doc.get('snippet','')}"
            )
        retrieved_context = "\n\n".join(context_lines)
        
        # Build JSON schema for structured output
        report_schema = {
            "type": "object",
            "properties": {
                "initial_workup": {"type": "string"},
                "risk_stratification": {"type": "string"},
                "risk_modifier": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                "surgical_thresholds": {"type": "string"},
                "imaging_surveillance": {"type": "string"},
                "lifestyle_guidelines": {"type": "string"},
                "pregnancy_peripartum": {"type": "string"},
                "genetic_counseling": {"type": "string"},
                "blood_pressure_recommendations": {"type": "string"},
                "medication_management": {"type": "string"},
                "gene_variant_interpretation": {"type": "string"},
                "references": {"type": "string"}
            },
            "required": [
                "initial_workup", "risk_stratification", "risk_modifier", 
                "surgical_thresholds", "imaging_surveillance", "lifestyle_guidelines",
                "pregnancy_peripartum", "genetic_counseling", "blood_pressure_recommendations",
                "medication_management", "gene_variant_interpretation", "references"
            ],
            "additionalProperties": False
        }
        
        # Generate report using OpenAI Responses API with structured output
        with st.spinner("Generating comprehensive report..."):
            try:
                # Prepare the full context
                full_context = (
                    "Patient Information:\n" + patient_context +
                    "\n\nRetrieved Medical Context:\n" + retrieved_context
                )
                
                response = self.client.responses.create(
                    model="gpt-4.1",
                    input=[
                        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                        {"role": "user", "content": full_context}
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "clinical_report",
                            "schema": report_schema,
                            "strict": True
                        }
                    }
                )
                
                # Parse response
                raw = response.output_text
                report_data = json.loads(raw)
                return report_data
                
            except Exception as e:
                st.error(f"Error generating report: {e}")
                return None
    
    def display_structured_report(self, report_data: Dict[str, Any]):
        """
        Display the structured report in Streamlit with proper formatting.
        
        Args:
            report_data: Dictionary containing report sections
        """
        if not report_data:
            st.error("No report data available")
            return
        
        # Generate and display Kaplan-Meier curve at the top
        st.subheader("📊 Risk Visualization")
        try:
            km_result = self.km_generator.generate_km_curve(st.session_state)
            if km_result:
                fig, interpretation = km_result
                self.km_generator.display_km_curve(fig, interpretation)
                st.divider()
        except Exception as e:
            st.warning(f"Could not generate Kaplan-Meier curve: {str(e)}")
        
        # Section mappings with icons
        sections = [
            ("🔬 Initial Workup", "initial_workup"),
            ("⚠️ Risk Stratification", "risk_stratification"),
            ("🏥 Surgical Thresholds", "surgical_thresholds"),
            ("🩻 Imaging Surveillance", "imaging_surveillance"),
            ("🏃 Lifestyle & Activity Guidelines", "lifestyle_guidelines"),
            ("🤰 Pregnancy/Peripartum", "pregnancy_peripartum"),
            ("👨‍👩‍👧‍👦 Genetic Counseling", "genetic_counseling"),
            ("💊 Blood Pressure Recommendations", "blood_pressure_recommendations"),
            ("💉 Medication Management", "medication_management"),
            ("🧬 Gene/Variant Interpretation", "gene_variant_interpretation")
        ]
        
        # Display risk modifier prominently
        if 'risk_modifier' in report_data:
            risk_mod = report_data['risk_modifier']
            if risk_mod < 1.0:
                risk_label = "Lower risk"
                risk_color = "🟢"
            elif risk_mod == 1.0:
                risk_label = "Standard risk"
                risk_color = "🟡"
            else:
                risk_label = "Higher risk"
                risk_color = "🔴"
            
            st.metric(
                label="Risk Modifier",
                value=f"{risk_mod:.1f}",
                delta=f"{risk_color} {risk_label}",
                delta_color="off"
            )
            st.divider()
        
        # Display sections in two columns
        col1, col2 = st.columns(2)
        
        # Split sections between columns
        for i, (title, key) in enumerate(sections):
            if key in report_data and report_data[key]:
                # Alternate between columns
                with col1 if i % 2 == 0 else col2:
                    st.subheader(title)
                    st.markdown(report_data[key])
                    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        
        # Display references
        if 'references' in report_data and report_data['references']:
            with st.expander("📚 References", expanded=False):
                st.markdown(report_data['references'])
    
    def export_report(self, report_data: Dict[str, Any], patient_info: Dict[str, Any]) -> str:
        """
        Export the report as a formatted markdown string.
        
        Args:
            report_data: Dictionary containing report sections
            patient_info: Patient demographic information
            
        Returns:
            Formatted markdown report
        """
        from datetime import datetime
        
        # Header
        export_text = f"""# AortaGPT Clinical Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Patient Information
- Age: {patient_info.get('age', 'Unknown')} years
- Sex: {patient_info.get('sex', 'Unknown')}
- Gene: {patient_info.get('gene', 'Unknown')}
- Variant: {patient_info.get('variant', 'Not specified')}
- Aortic Root: {patient_info.get('root_diameter', 0)} mm
- Ascending Aorta: {patient_info.get('ascending_diameter', 0)} mm

---

"""
        
        # Section headings
        section_titles = {
            "initial_workup": "## Initial Workup",
            "risk_stratification": "## Risk Stratification",
            "surgical_thresholds": "## Surgical Thresholds",
            "imaging_surveillance": "## Imaging Surveillance",
            "lifestyle_guidelines": "## Lifestyle & Activity Guidelines",
            "pregnancy_peripartum": "## Pregnancy/Peripartum",
            "genetic_counseling": "## Genetic Counseling",
            "blood_pressure_recommendations": "## Blood Pressure Recommendations",
            "medication_management": "## Medication Management",
            "gene_variant_interpretation": "## Gene/Variant Interpretation"
        }
        
        # Add risk modifier
        if 'risk_modifier' in report_data:
            export_text += f"**Risk Modifier: {report_data['risk_modifier']:.1f}**\n\n"
        
        # Add sections
        for key, title in section_titles.items():
            if key in report_data and report_data[key]:
                export_text += f"{title}\n{report_data[key]}\n\n"
        
        # Add references
        if 'references' in report_data:
            export_text += f"## References\n{report_data['references']}\n"
        
        return export_text
    
    def save_report_to_session(self, report: str):
        """Save the generated report to session state for later access."""
        if 'generated_reports' not in st.session_state:
            st.session_state.generated_reports = []
        
        report_entry = {
            'timestamp': st.session_state.get('last_report_timestamp', ''),
            'report': report,
            'patient_params': {
                'age': st.session_state.get('age', 0),
                'sex': st.session_state.get('sex', 'Other'),
                'gene': st.session_state.get('gene', 'Other'),
                'variant': st.session_state.get('variant', ''),
                'root_diameter': st.session_state.get('root_diameter', 0),
                'ascending_diameter': st.session_state.get('ascending_diameter', 0)
            }
        }
        
        st.session_state.generated_reports.append(report_entry)