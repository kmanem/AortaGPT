"""
Text interpretation module for AortaGPT.
Handles parsing of free-text patient descriptions into structured parameters.
"""
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import streamlit as st
import json
from openai import OpenAI


class InterpretationState(Enum):
    """States for the text interpretation process."""
    IDLE = "idle"
    CONFIRMING = "confirming"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class PatientParameters:
    """Structured patient parameters extracted from text."""
    age: int = 0
    sex: str = "Other"
    gene: str = "Other"
    custom_gene: str = ""
    variant: str = ""
    root_diameter: float = 0.0
    ascending_diameter: float = 0.0
    z_score: float = 0.0
    meds: List[str] = None
    other_relevant_details: str = ""
    
    # Clinical history flags
    diagnosis_aortic_aneurysm: bool = False
    family_history_aortic: bool = False
    diagnosis_other_aneurysm: bool = False
    family_history_other_aneurysm: bool = False
    diagnosis_hcm: bool = False
    family_history_hcm: bool = False
    diagnosis_dcm: bool = False
    family_history_dcm: bool = False
    diagnosis_long_qt: bool = False
    family_history_long_qt: bool = False
    diagnosis_dyslipidemia: bool = False
    family_history_dyslipidemia: bool = False
    marfanoid_features: bool = False
    loeys_dietz_features: bool = False
    ehlers_danlos_features: bool = False
    pregnant_or_considering: bool = False
    
    def __post_init__(self):
        if self.meds is None:
            self.meds = []
    
    def to_session_state(self):
        """Update Streamlit session state with these parameters."""
        # Basic fields
        st.session_state['age'] = self.age
        st.session_state['sex'] = self.sex
        st.session_state['gene'] = self.gene
        st.session_state['custom_gene'] = self.custom_gene
        st.session_state['variant'] = self.variant
        st.session_state['root_diameter'] = self.root_diameter
        st.session_state['ascending_diameter'] = self.ascending_diameter
        st.session_state['z_score'] = self.z_score
        st.session_state['meds'] = self.meds
        st.session_state['other_relevant_details'] = self.other_relevant_details
        
        # Clinical history flags
        st.session_state['Diagnosis of Aortic Aneurysm and/or Dissection'] = self.diagnosis_aortic_aneurysm
        st.session_state['Family History of Aortic Aneurysm or Dissection'] = self.family_history_aortic
        st.session_state['Diagnosis of Aneurysm in Other Arteries'] = self.diagnosis_other_aneurysm
        st.session_state['Family History of Aneurysm in Other Arteries'] = self.family_history_other_aneurysm
        st.session_state['Diagnosis of Hypertrophic Cardiomyopathy'] = self.diagnosis_hcm
        st.session_state['Family History of Hypertrophic Cardiomyopathy'] = self.family_history_hcm
        st.session_state['Diagnosis of Dilated Cardiomyopathy'] = self.diagnosis_dcm
        st.session_state['Family History of Dilated Cardiomyopathy'] = self.family_history_dcm
        st.session_state['Diagnosis of Long QT Syndrome'] = self.diagnosis_long_qt
        st.session_state['Family History of Long QT Syndrome'] = self.family_history_long_qt
        st.session_state['Diagnosis of Dyslipidemia'] = self.diagnosis_dyslipidemia
        st.session_state['Family History of Dyslipidemia'] = self.family_history_dyslipidemia
        st.session_state['Marfanoid Features Present'] = self.marfanoid_features
        st.session_state['Loeys-Dietz Features Present'] = self.loeys_dietz_features
        st.session_state['Ehlers-Danlos Features Present'] = self.ehlers_danlos_features
        st.session_state['Currently Pregnant or Considering Pregnancy'] = self.pregnant_or_considering
    
    @classmethod
    def from_json_response(cls, json_data: Dict[str, Any]) -> 'PatientParameters':
        """Create PatientParameters from OpenAI JSON response."""
        return cls(
            age=json_data.get('age', 0),
            sex=json_data.get('sex', 'Other'),
            gene=json_data.get('gene', 'Other'),
            custom_gene=json_data.get('custom_gene', ''),
            variant=json_data.get('variant', ''),
            root_diameter=json_data.get('root_diameter', 0.0),
            ascending_diameter=json_data.get('ascending_diameter', 0.0),
            z_score=json_data.get('z_score', 0.0),
            meds=json_data.get('meds', []),
            other_relevant_details=json_data.get('other_relevant_details', ''),
            # Clinical history mappings
            diagnosis_aortic_aneurysm=json_data.get('Diagnosis of Aortic Aneurysm and/or Dissection', False),
            family_history_aortic=json_data.get('Family History of Aortic Aneurysm or Dissection', False),
            diagnosis_other_aneurysm=json_data.get('Diagnosis of Aneurysm in Other Arteries', False),
            family_history_other_aneurysm=json_data.get('Family History of Aneurysm in Other Arteries', False),
            diagnosis_hcm=json_data.get('Diagnosis of Hypertrophic Cardiomyopathy', False),
            family_history_hcm=json_data.get('Family History of Hypertrophic Cardiomyopathy', False),
            diagnosis_dcm=json_data.get('Diagnosis of Dilated Cardiomyopathy', False),
            family_history_dcm=json_data.get('Family History of Dilated Cardiomyopathy', False),
            diagnosis_long_qt=json_data.get('Diagnosis of Long QT Syndrome', False),
            family_history_long_qt=json_data.get('Family History of Long QT Syndrome', False),
            diagnosis_dyslipidemia=json_data.get('Diagnosis of Dyslipidemia', False),
            family_history_dyslipidemia=json_data.get('Family History of Dyslipidemia', False),
            marfanoid_features=json_data.get('Marfanoid Features Present', False),
            loeys_dietz_features=json_data.get('Loeys-Dietz Features Present', False),
            ehlers_danlos_features=json_data.get('Ehlers-Danlos Features Present', False),
            pregnant_or_considering=json_data.get('Currently Pregnant or Considering Pregnancy', False)
        )


class TextInterpretationManager:
    """Manages the text interpretation workflow."""
    
    def __init__(self, client: OpenAI, gene_options: List[str], clinical_options: List[str]):
        self.client = client
        self.gene_options = gene_options
        self.clinical_options = clinical_options
        
        # Initialize state if not exists
        if 'interpretation_state' not in st.session_state:
            st.session_state.interpretation_state = InterpretationState.IDLE
        if 'interpretation_text' not in st.session_state:
            st.session_state.interpretation_text = ""
        if 'interpretation_error' not in st.session_state:
            st.session_state.interpretation_error = None
    
    @property
    def state(self) -> InterpretationState:
        """Get current interpretation state."""
        return st.session_state.interpretation_state
    
    @state.setter
    def state(self, value: InterpretationState):
        """Set interpretation state."""
        st.session_state.interpretation_state = value
    
    def reset(self):
        """Reset to idle state."""
        self.state = InterpretationState.IDLE
        st.session_state.interpretation_text = ""
        st.session_state.interpretation_error = None
    
    def start_confirmation(self, text: str):
        """Start the confirmation process."""
        st.session_state.interpretation_text = text
        self.state = InterpretationState.CONFIRMING
    
    def cancel_confirmation(self):
        """Cancel the confirmation and return to idle."""
        self.reset()
    
    def build_json_schema(self, description: str) -> Dict[str, Any]:
        """Build the JSON schema for structured output."""
        # Narrow gene choices based on description
        candidates = [g for g in self.gene_options if g != "Other" and g.lower() in description.lower()]
        gene_enum = candidates + ["Other"] if candidates else self.gene_options.copy()
        
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
        
        # Add clinical history booleans
        for opt in self.clinical_options:
            schema["properties"][opt] = {"type": "boolean"}
            schema["required"].append(opt)
        
        # Add free-text field
        schema["properties"]["other_relevant_details"] = {"type": "string"}
        schema["required"].append("other_relevant_details")
        
        return schema
    
    def interpret_text(self) -> Optional[PatientParameters]:
        """
        Interpret the stored text and return PatientParameters.
        Updates state during the process.
        """
        if self.state != InterpretationState.CONFIRMING:
            return None
        
        self.state = InterpretationState.PROCESSING
        description = st.session_state.interpretation_text
        
        try:
            # Build system and user messages
            system_msg = (
                "You are a helpful assistant that extracts patient parameters from a free-form description. "
                "Also populate 'other_relevant_details' with any additional clinically relevant details."
            )
            
            # Call OpenAI API (keeping the exact current syntax)
            response = self.client.responses.create(
                model="gpt-4.1-nano",
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": description}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "patient_config",
                        "schema": self.build_json_schema(description),
                        "strict": True
                    }
                }
            )
            
            # Parse response
            raw = response.output_text
            print(raw)  # Keep for debugging
            json_data = json.loads(raw)
            
            # Create PatientParameters
            params = PatientParameters.from_json_response(json_data)
            
            # Update session state
            params.to_session_state()
            
            # Success!
            self.state = InterpretationState.COMPLETED
            return params
            
        except Exception as e:
            st.session_state.interpretation_error = str(e)
            self.state = InterpretationState.ERROR
            return None
    
    def render_ui(self):
        """Render the text interpretation UI in the sidebar."""
        with st.expander("🗣 Set Parameters from Text", expanded=(self.state != InterpretationState.IDLE)):
            
            if self.state == InterpretationState.IDLE:
                # Initial state - show text area and button
                text = st.text_area("Describe the patient", key="patient_description_input")
                if st.button("Interpret & Apply Parameters") and text:
                    self.start_confirmation(text)
                    st.rerun()
            
            elif self.state == InterpretationState.CONFIRMING:
                # Show confirmation
                st.info("📋 **Before proceeding, ensure your description includes:**")
                st.markdown("""
                - Patient demographics (age, sex)
                - Genetic information (gene, variant if known)
                - Aortic measurements
                - Current medications
                - Relevant medical/family history
                - Any symptoms or concerns
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Proceed", use_container_width=True, type="primary"):
                        params = self.interpret_text()
                        if params:
                            st.success("✨ Parameters applied successfully!")
                            # Only rerun after successful interpretation
                            st.rerun()
                
                with col2:
                    if st.button("↩️ Go Back", use_container_width=True):
                        self.cancel_confirmation()
                        st.rerun()
            
            elif self.state == InterpretationState.PROCESSING:
                # Show spinner
                st.spinner("Interpreting parameters...")
            
            elif self.state == InterpretationState.COMPLETED:
                # Show success and reset button
                st.success("✅ Parameters have been applied!")
                if st.button("Interpret Another Description"):
                    self.reset()
                    st.rerun()
            
            elif self.state == InterpretationState.ERROR:
                # Show error
                st.error(f"❌ Error: {st.session_state.interpretation_error}")
                if st.button("Try Again"):
                    self.reset()
                    st.rerun()