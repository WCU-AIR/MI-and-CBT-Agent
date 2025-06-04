import os
import random

def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Case study file not found: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Failed to read case study: {e}")


#Reads a patient profile and returns it as a string
def load_patient_profile(filepath: str) -> str:
    return read_file(filepath)

def load_random_case_profile(directory='Patient_Profiles'):
    files = [f for f in os.listdir(directory)]
    if not files:
        raise FileNotFoundError("No patient profiles found in the directory.")

    random_file = random.choice(files)
    full_path = os.path.join(directory, random_file)
    return load_patient_profile(full_path)

def load_prompt(prompt_name: str) -> str:

    if prompt_name == 'Clinical Summary':
        filepath = 'Prompts/clinical_summary_prompt.txt'
    elif prompt_name == 'Virtual Patient':
        filepath = 'Prompts/virtual_patient_prompt.txt'
    elif prompt_name == 'Verification':
        filepath = 'Prompts/patient_profile_verification_prompt.txt'
    elif prompt_name == 'Feedback':
        filepath = 'Prompts/feedback_agent_prompt.txt'
    elif prompt_name == 'STACER Prompt':
        filepath = 'Prompts/STACER_form_prompt.txt'
    else:
        filepath = ''

    return read_file(filepath)

def load_context(context_name: str) -> str:

    if context_name == 'Feedback Criteria':
        filepath = 'Context/feedback_criteria.txt'
    elif context_name == 'COD STACER':
        filepath = 'Context/COD_STACER.txt'
    elif context_name == 'FOD STACER':
        filepath = 'Context/FOD_STACER.txt'
    else:
        filepath = ''

    return read_file(filepath)