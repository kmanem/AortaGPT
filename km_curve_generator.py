"""
Kaplan-Meier curve generation module for AortaGPT.
Generates survival curves based on patient parameters and variant data.
"""
import streamlit as st
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import json


class KMCurveGenerator:
    """Generates Kaplan-Meier curves using GPT-4 with variant-specific data."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        
    def generate_km_curve(self, session_state: Dict[str, Any]) -> Optional[Tuple[plt.Figure, str]]:
        """
        Generate a Kaplan-Meier curve for the patient using GPT-4.
        
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
        
        # Build the prompt with patient-specific data
        patient_context = f"""
Patient Information:
- Gene: {gene}
- Variant: {variant}
- Age: {age} years
- Sex: {sex}
- Clinical Significance: {variant_details.get('clinical_significance', 'Unknown') if variant_details else 'Unknown'}
- Review Status: {variant_details.get('review_status', 'Unknown') if variant_details else 'Unknown'}
"""
        
        # System prompt for KM curve generation
        km_system_prompt = """Your job is to generate Python code for a Kaplan-Meier curve using the patient data provided. 
The code should:
1. Create a clinically accurate survival curve based on the gene/variant
2. Use realistic event data based on known literature about this condition
3. Include appropriate censoring patterns
4. Mark the patient's current age on the curve
5. Add a general population reference line for comparison
6. Return ONLY executable Python code that creates a matplotlib figure

Use this template structure:
```python
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter

# Event data based on literature for this gene/variant
ages_of_event = [...]  # Ages when aortic events occurred
censored = [...]  # 0 = event occurred, 1 = censored

# Fit Kaplan-Meier curve
kmf = KaplanMeierFitter()
kmf.fit(ages_of_event, event_observed=[1 if c==0 else 0 for c in censored])

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2.5, color="crimson", label=f"{gene} {variant}")

# Add general population reference
x_pop = np.linspace(0, 90, 100)
y_pop = 1 - (1 - np.exp(-0.00001 * x_pop**1.5))  # General population curve
ax.plot(x_pop, y_pop, '--', color='gray', alpha=0.7, linewidth=2, label="General Population")

# Mark patient's current age
patient_age = {age}
if patient_age <= max(ages_of_event):
    surv_prob = kmf.survival_function_at_times(patient_age).iloc[0]
    ax.plot(patient_age, surv_prob, 'ko', markersize=12)
    ax.annotate(f'Current Age: {patient_age}', 
                xy=(patient_age, surv_prob),
                xytext=(patient_age+5, surv_prob+0.05),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Styling
ax.set_xlabel("Age (years)", fontsize=12, fontweight='bold')
ax.set_ylabel("Event-Free Survival", fontsize=12, fontweight='bold')
ax.set_title(f"Kaplan-Meier Curve: {gene} {variant}", fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='lower left', fontsize=11)
ax.set_xlim(0, 90)
ax.set_ylim(0, 1.05)

# Add risk annotations
ax.text(0.02, 0.02, f"Patient: {age}y {sex}", transform=ax.transAxes, 
        fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
```

Adjust the event ages based on the specific gene/variant severity. For example:
- FBN1: Events typically 30-60 years
- TGFBR1/2: Earlier events, 20-50 years  
- COL3A1: Very early events, 20-40 years
- ACTA2: Variable, 25-55 years"""

        try:
            # Make the API call
            with st.spinner("Generating Kaplan-Meier curve..."):
                response = self.client.responses.create(
                    model="gpt-4.1",
                    input=[
                        {"role": "system", "content": km_system_prompt},
                        {"role": "user", "content": patient_context}
                    ],
                    text={"format": {"type": "text"}},
                    temperature=0.7,
                    max_output_tokens=2048
                )
                
                # Extract the Python code from response
                code_text = response.output_text
                
                # Clean up the code (remove markdown backticks if present)
                if "```python" in code_text:
                    code_text = code_text.split("```python")[1].split("```")[0]
                elif "```" in code_text:
                    code_text = code_text.split("```")[1].split("```")[0]
                
                # Execute the generated code
                namespace = {}
                exec(code_text, namespace)
                
                # Get the figure from the namespace
                fig = plt.gcf()  # Get current figure
                
                # Generate interpretation
                interpretation = self._generate_interpretation(gene, variant, age, sex)
                
                return fig, interpretation
                
        except Exception as e:
            st.error(f"Error generating Kaplan-Meier curve: {str(e)}")
            # Fallback to simple curve generation
            return self._generate_fallback_curve(gene, variant, age, sex)
    
    def _generate_interpretation(self, gene: str, variant: str, age: int, sex: str) -> str:
        """Generate interpretation text for the KM curve."""
        interpretations = {
            "FBN1": f"This Kaplan-Meier curve shows the cumulative risk of aortic events for patients with {gene} mutations. The median event-free survival is typically around 45-50 years without treatment.",
            "TGFBR1": f"Patients with {gene} mutations (Loeys-Dietz syndrome) show earlier aortic events. Close surveillance and early intervention are critical.",
            "TGFBR2": f"This curve demonstrates the aggressive nature of {gene}-related aortopathy. Most events occur before age 40 without preventive surgery.",
            "COL3A1": f"Vascular Ehlers-Danlos syndrome ({gene}) carries the highest risk. Arterial events can occur at young ages throughout the arterial tree.",
            "ACTA2": f"The {gene} mutation shows variable penetrance. Risk increases significantly after age 30, with additional cerebrovascular risks.",
        }
        
        default = f"This Kaplan-Meier curve estimates the cumulative risk of aortic events for patients with {gene} mutations based on available literature data."
        
        base_interpretation = interpretations.get(gene, default)
        
        # Add age-specific commentary
        if age < 30:
            age_comment = f" At age {age}, the patient is in a lower-risk period but requires regular surveillance."
        elif age < 50:
            age_comment = f" At age {age}, the patient is entering a higher-risk period. Intensified surveillance is recommended."
        else:
            age_comment = f" At age {age}, the patient has passed the typical high-risk period but continued monitoring is essential."
            
        return base_interpretation + age_comment
    
    def _generate_fallback_curve(self, gene: str, variant: str, age: int, sex: str) -> Tuple[plt.Figure, str]:
        """Generate a simple fallback KM curve if API fails."""
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.linspace(0, 90, 1000)
        
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
        
        # Calculate survival curve
        y = np.exp(-baseline * x**power)
        
        # Plot main curve
        ax.plot(x, y, color='crimson', linewidth=3, label=f"{gene} {variant if variant else ''}")
        
        # Add confidence interval
        ci_upper = np.minimum(y + 0.1, 1.0)
        ci_lower = np.maximum(y - 0.1, 0.0)
        ax.fill_between(x, ci_lower, ci_upper, color='crimson', alpha=0.2)
        
        # Add general population reference
        y_pop = np.exp(-0.00001 * x**1.5)
        ax.plot(x, y_pop, '--', color='gray', alpha=0.7, linewidth=2, label="General Population")
        
        # Mark current age
        if 0 < age < 90:
            current_survival = np.exp(-baseline * age**power)
            ax.plot(age, current_survival, 'ko', markersize=12)
            ax.annotate(f'Current Age: {age}', 
                       xy=(age, current_survival),
                       xytext=(age+5, current_survival+0.05),
                       arrowprops=dict(facecolor='black', shrink=0.05),
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Styling
        ax.set_xlabel("Age (years)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Event-Free Survival", fontsize=12, fontweight='bold')
        ax.set_title(f"Estimated Aortic Event Risk: {gene} {variant if variant else ''}", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='lower left', fontsize=11)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1.05)
        
        # Add patient info
        ax.text(0.02, 0.02, f"Patient: {age}y {sex}", transform=ax.transAxes, 
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
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