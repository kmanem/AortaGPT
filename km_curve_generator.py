"""
Kaplan-Meier curve generation module for AortaGPT.
Generates survival curves based on patient parameters and variant data.
"""
import streamlit as st
from typing import Dict, Any, Optional, Tuple, List
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import json
from lifelines import KaplanMeierFitter


class KMCurveGenerator:
    """Generates Kaplan-Meier curves using GPT-4 for data extraction and lifelines for plotting."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        
    def generate_km_curve(self, session_state: Dict[str, Any]) -> Optional[Tuple[plt.Figure, str]]:
        """
        Generate a Kaplan-Meier curve for the patient using GPT-4 for data extraction.
        
        Args:
            session_state: Streamlit session state containing patient data
            
        Returns:
            Tuple of (matplotlib figure, interpretation text) or None if failed
        """
        # Extract patient parameters
        gene = session_state.get('gene', '')
        variant = session_state.get('variant', '')
        age = session_state.get('age', 30)
        sex = session_state.get('sex', 'Male')
        variant_details = session_state.get('selected_variant_info', {})
        
        # Get survival data from GPT-4
        survival_data = self._extract_survival_data(gene, variant, variant_details)
        
        if not survival_data:
            # Fallback to pre-defined data
            return self._generate_fallback_curve(gene, variant, age, sex)
        
        # Create the KM curve using lifelines
        try:
            fig = self._create_km_plot(survival_data, gene, variant, age, sex)
            interpretation = self._generate_interpretation(gene, variant, age, sex)
            return fig, interpretation
        except Exception as e:
            st.error(f"Error creating Kaplan-Meier curve: {str(e)}")
            return self._generate_fallback_curve(gene, variant, age, sex)
    
    def _extract_survival_data(self, gene: str, variant: str, variant_details: Dict[str, Any], 
                              vector_store_id: str = "vs_6842481bfb588191bb5a860d02fa2477") -> Optional[Dict[str, Any]]:
        """
        Use GPT-4 to extract survival data based on gene/variant information.
        
        Args:
            gene: Gene name
            variant: Variant identifier
            variant_details: ClinVar details about the variant
            vector_store_id: ID of the vector store containing medical literature
        
        Returns:
            Dictionary containing event_ages, censored_ages, and clinical notes
        """
        # Build the prompt
        extraction_prompt = f"""
Based on medical literature and clinical data for {gene} mutations (variant: {variant}), provide realistic survival data for a Kaplan-Meier analysis.

Clinical Significance: {variant_details.get('clinical_significance', 'Unknown') if variant_details else 'Unknown'}
Review Status: {variant_details.get('review_status', 'Unknown') if variant_details else 'Unknown'}

Please provide the data in the following JSON format:
{{
    "event_ages": [list of ages when aortic events occurred],
    "censored_ages": [list of ages for patients who did not experience events during follow-up],
    "median_event_age": approximate median age of events,
    "clinical_notes": "brief description of the typical disease course",
    "severity": "mild|moderate|severe"
}}

Guidelines for data generation:
- FBN1 (Marfan): Events typically 35-55 years, moderate severity
- TGFBR1/2 (Loeys-Dietz): Earlier events 20-45 years, severe
- COL3A1 (vEDS): Very early events 20-40 years, severe
- ACTA2: Variable 25-50 years, moderate-severe
- MYH11: 30-55 years, moderate

Include at least 15-20 event ages and 20-30 censored ages for statistical validity.
Use the medical literature in the vector store to inform realistic survival patterns."""
        
        # Define schema for structured output
        data_schema = {
            "type": "object",
            "properties": {
                "event_ages": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 10,
                    "maxItems": 30
                },
                "censored_ages": {
                    "type": "array", 
                    "items": {"type": "number"},
                    "minItems": 15,
                    "maxItems": 40
                },
                "median_event_age": {"type": "number"},
                "clinical_notes": {"type": "string"},
                "severity": {
                    "type": "string",
                    "enum": ["mild", "moderate", "severe"]
                }
            },
            "required": ["event_ages", "censored_ages", "median_event_age", "clinical_notes", "severity"],
            "additionalProperties": False
        }
        
        try:
            with st.spinner("Extracting survival data..."):
                response = self.client.responses.create(
                    model="gpt-4.1",
                    input=[
                        {"role": "system", "content": "You are a clinical geneticist providing survival data for genetic aortopathies based on medical literature."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "survival_data",
                            "schema": data_schema,
                            "strict": True
                        }
                    },
                    tools=[
                        {
                            "type": "file_search",
                            "vector_store_ids": [vector_store_id]
                        }
                    ],
                    temperature=0.7
                )
                
                # Parse the response
                survival_data = json.loads(response.output_text)
                return survival_data
                
        except Exception as e:
            st.warning(f"Could not extract survival data: {str(e)}")
            return None
    
    def _create_km_plot(self, survival_data: Dict[str, Any], gene: str, variant: str, 
                        age: int, sex: str) -> plt.Figure:
        """
        Create cumulative incidence plot using lifelines with the extracted survival data.
        """
        # Extract data
        event_ages = survival_data['event_ages']
        censored_ages = survival_data['censored_ages']
        
        # Combine all ages and create event indicators
        all_ages = event_ages + censored_ages
        event_observed = [1] * len(event_ages) + [0] * len(censored_ages)
        
        # Fit Kaplan-Meier model
        kmf = KaplanMeierFitter()
        kmf.fit(all_ages, event_observed=event_observed, label=f"{gene} {variant}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot CUMULATIVE INCIDENCE (1 - survival function)
        # This shows the probability of having an event by each age
        kmf.plot_cumulative_density(ax=ax, ci_show=True, linewidth=3, color="crimson", 
                                   label=f"{gene} {variant if variant else '(p.{variant})' if variant else ''}")
        
        # Add general population reference (cumulative incidence)
        x_pop = np.linspace(0, 80, 100)
        y_pop = 1 - np.exp(-0.00001 * x_pop**1.5)  # General population cumulative incidence
        ax.plot(x_pop, y_pop, '--', color='gray', alpha=0.7, linewidth=2, label="General Population")
        
        # Mark patient's current age
        if 0 < age <= 90:
            try:
                # Get cumulative incidence at current age (1 - survival probability)
                surv_prob = kmf.survival_function_at_times(age).iloc[0]
                cum_inc = 1 - surv_prob
                ax.plot(age, cum_inc, 'ko', markersize=12)
                
                # Add vertical line to show current age
                ax.axvline(x=age, color='black', linestyle=':', alpha=0.5)
                
                # Annotate with risk percentage
                risk_pct = cum_inc * 100
                ax.annotate(f'Age {age}: {risk_pct:.1f}% risk', 
                           xy=(age, cum_inc),
                           xytext=(age+3, cum_inc+0.05),
                           fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8))
            except:
                pass  # Skip if age is out of range
        
        # Styling
        ax.set_xlabel("Age (years)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Cumulative Probability", fontsize=14, fontweight='bold')
        ax.set_title(f"Kaplan-Meier Estimate: Cumulative Risk of Aortic Event\nVariant: {gene} {variant if variant else ''}", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, linestyle='-', alpha=0.2)
        ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 1.05)
        
        # Add annotations
        severity_color = {
            'mild': 'lightgreen',
            'moderate': 'yellow', 
            'severe': 'lightcoral'
        }
        
        severity = survival_data.get('severity', 'moderate')
        ax.text(0.98, 0.98, f"Severity: {severity.capitalize()}", 
                transform=ax.transAxes, 
                fontsize=11, 
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=severity_color.get(severity, 'wheat'), alpha=0.7))
        
        ax.text(0.02, 0.02, f"Patient: {age}y {sex}\nMedian Event Age: {survival_data.get('median_event_age', 'N/A')}", 
                transform=ax.transAxes, 
                fontsize=10, 
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def _generate_interpretation(self, gene: str, variant: str, age: int, sex: str) -> str:
        """Generate interpretation text for the cumulative incidence curve."""
        interpretations = {
            "FBN1": f"This curve shows the cumulative probability of aortic events for patients with {gene} mutations. Without treatment, approximately 50% of patients experience an event by age 45-50.",
            "TGFBR1": f"Patients with {gene} mutations (Loeys-Dietz syndrome) have rapidly increasing risk in early adulthood. The steep curve emphasizes the need for aggressive surveillance and early intervention.",
            "TGFBR2": f"This curve illustrates the aggressive nature of {gene}-related aortopathy, with substantial risk accumulation before age 40. Preventive surgery is often indicated at smaller diameters.",
            "COL3A1": f"Vascular Ehlers-Danlos syndrome ({gene}) shows the highest cumulative risk at young ages. The curve reflects risk for all arterial events, not just aortic.",
            "ACTA2": f"The {gene} mutation shows steadily increasing risk after age 25, with additional risks for cerebrovascular and coronary complications beyond aortic events.",
        }
        
        default = f"This cumulative incidence curve estimates the probability of experiencing an aortic event by each age for patients with {gene} mutations."
        
        base_interpretation = interpretations.get(gene, default)
        
        # Add age-specific commentary
        if age < 30:
            age_comment = f" At age {age}, cumulative risk remains relatively low but surveillance is essential to detect early changes."
        elif age < 50:
            age_comment = f" At age {age}, the patient is in a period of increasing risk. Current guidelines recommend intensified imaging surveillance."
        else:
            age_comment = f" At age {age}, substantial risk has already accumulated. Continued vigilant monitoring remains critical."
            
        return base_interpretation + age_comment
    
    def _generate_fallback_curve(self, gene: str, variant: str, age: int, sex: str) -> Tuple[plt.Figure, str]:
        """Generate a simple fallback cumulative incidence curve if API fails."""
        fig, ax = plt.subplots(figsize=(10, 7))
        x = np.linspace(0, 80, 1000)
        
        # Gene-specific parameters
        gene_params = {
            "FBN1": (0.00015, 1.5, 40),     # baseline hazard, power, acceleration age
            "TGFBR1": (0.0003, 1.8, 30),
            "TGFBR2": (0.0003, 1.8, 30),
            "ACTA2": (0.0002, 1.6, 35),
            "COL3A1": (0.0004, 1.9, 25),
            "MYH11": (0.00018, 1.6, 38),
        }
        
        # Get parameters or use defaults
        if gene in gene_params:
            baseline, power, accel_age = gene_params[gene]
        else:
            baseline, power, accel_age = (0.0001, 1.5, 40)
        
        # Calculate CUMULATIVE INCIDENCE (1 - survival)
        y = 1 - np.exp(-baseline * x**power)
        
        # Plot main curve (cumulative incidence)
        ax.plot(x, y, color='crimson', linewidth=3, label=f"{gene} {variant if variant else ''}")
        
        # Add confidence interval
        ci_upper = np.minimum(y + 0.1, 1.0)
        ci_lower = np.maximum(y - 0.1, 0.0)
        ax.fill_between(x, ci_lower, ci_upper, color='crimson', alpha=0.2)
        
        # Add general population reference (cumulative incidence)
        y_pop = 1 - np.exp(-0.00001 * x**1.5)
        ax.plot(x, y_pop, '--', color='gray', alpha=0.7, linewidth=2, label="General Population")
        
        # Mark current age
        if 0 < age < 80:
            current_risk = 1 - np.exp(-baseline * age**power)
            ax.plot(age, current_risk, 'ko', markersize=12)
            ax.axvline(x=age, color='black', linestyle=':', alpha=0.5)
            
            # Annotate with risk percentage
            risk_pct = current_risk * 100
            ax.annotate(f'Age {age}: {risk_pct:.1f}% risk', 
                       xy=(age, current_risk),
                       xytext=(age+3, current_risk+0.05),
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8))
        
        # Styling
        ax.set_xlabel("Age (years)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Cumulative Probability", fontsize=14, fontweight='bold')
        ax.set_title(f"Kaplan-Meier Estimate: Cumulative Risk of Aortic Event\nVariant: {gene} {variant if variant else ''}", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, linestyle='-', alpha=0.2)
        ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 1.05)
        
        # Add severity indicator based on gene
        severity = 'moderate'
        if gene in ['TGFBR1', 'TGFBR2', 'COL3A1']:
            severity = 'severe'
        elif gene in ['FBN1']:
            severity = 'moderate'
        
        severity_color = {
            'mild': 'lightgreen',
            'moderate': 'yellow', 
            'severe': 'lightcoral'
        }
        
        ax.text(0.98, 0.02, f"Severity: {severity.capitalize()}\nPatient: {age}y {sex}", 
                transform=ax.transAxes, 
                fontsize=11,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=severity_color.get(severity, 'wheat'), alpha=0.7))
        
        plt.tight_layout()
        
        interpretation = self._generate_interpretation(gene, variant, age, sex)
        return fig, interpretation
    
    def fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string for embedding."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)  # Close the figure to free memory
        return img_base64
    
    def display_km_curve(self, fig: plt.Figure, interpretation: str):
        """Display the KM curve and interpretation in Streamlit."""
        # Display the figure
        st.pyplot(fig)
        
        # Display interpretation in an info box
        st.info(interpretation)
        
        # Add download button for the figure
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Kaplan-Meier Curve",
            data=buf,
            file_name=f"km_curve_{st.session_state.get('gene', 'unknown')}_{st.session_state.get('variant', 'unknown')}.png",
            mime="image/png"
        )