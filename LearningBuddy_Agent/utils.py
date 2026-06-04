import os
import random
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import logging

# Import main class and constants from learning_buddy module



###############################################################
###                    FileManager Class                    ###
###############################################################

class FileManager:
    """Handles all file I/O operations for LearningBuddy."""

    def __init__(self, mode: int):
        """
        Initialize FileManager with the current mode.

        Args:
            mode: One of STACER_MODE, MI_MODE, or CBT_MODE
        """
        self.mode = mode

    def read_file(self, filepath: str) -> str:
        """
        Read and return contents of a file.

        Args:
            filepath: Path to the file to read

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If reading fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

    def load_patient_profile(self, filepath: str) -> str:
        """
        Load and return a specific patient profile.

        Args:
            filepath: Path to the patient profile file

        Returns:
            Patient profile content as string
        """
        return self.read_file(filepath)

    def load_random_case_profile(self, directory: str = 'Patient_Profiles') -> str:
        from LearningBuddy_Agent import STACER_MODE,  MI_MODE, CBT_MODE
        """
        Load a random patient profile from the appropriate mode directory.

        Args:
            directory: Base directory containing patient profiles

        Returns:
            Random patient profile content as string

        Raises:
            FileNotFoundError: If no profiles found in directory
        """
        # Select Patients available in the current mode
        if self.mode == STACER_MODE or self.mode == CBT_MODE:
            directory += "/CBT"
        elif self.mode == MI_MODE:
            directory += "/MI"
        # Add more modes ...


        print("Dir:",directory)
        files = [f for f in os.listdir(directory)]
        if not files:
            raise FileNotFoundError("No patient profiles found in the directory.")
        random_file = random.choice(files)
        full_path = os.path.join(directory, random_file)
        print("FullPath:", full_path)
        return self.load_patient_profile(full_path)

    def calculate_session_duration(self, start_time_iso: str) -> float:
        """
        Calculate session duration in minutes.

        Args:
            start_time_iso: ISO format start time string

        Returns:
            Duration in minutes as float
        """
        try:
            start_time = datetime.fromisoformat(start_time_iso)
            duration = datetime.now() - start_time
            return duration.total_seconds() / 60
        except:
            return 0.0

    def save_session_data(
            self,
            case_study: str,
            stacer: str,
            detailed_feedback: str,
            transcript: str,
            presentation: str,
            mode: int,
            base_dir: str = "Saved_Sessions",
            session_id: str = None
    ) -> str:
        """
        Save all session data to timestamped folder.

        Args:
            case_study: Patient case study content
            stacer: Completed CBT/evaluation form
            detailed_feedback: Detailed feedback text
            transcript: Formatted interview transcript
            presentation: Formatted presentation transcript
            mode: Current mode (STACER_MODE, MI_MODE, or CBT_MODE)
            base_dir: Base directory for saving sessions
            session_id: Optional session identifier with metadata (patient number, student type)

        Returns:
            Path to the created folder
        """
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Create timestamped subfolder with session_id if provided
        if session_id:
            # Use the full session_id which already includes timestamp and metadata
            # Format: session_20251105_175800_patient_1_good_student
            # Convert to: session_export_2025-11-05_17-58_patient_1_good_student
            if session_id.startswith("session_"):
                # Extract timestamp and metadata parts
                parts = session_id.split("_", 2)  # ['session', '20251105', '175800_patient_1_good_student']
                if len(parts) >= 3:
                    date_part = parts[1]  # '20251105'
                    time_and_meta = parts[2]  # '175800_patient_1_good_student'

                    # Split time from metadata
                    time_meta_parts = time_and_meta.split("_", 1)  # ['175800', 'patient_1_good_student']
                    time_part = time_meta_parts[0]  # '175800'
                    metadata_part = time_meta_parts[1] if len(time_meta_parts) > 1 else ""  # 'patient_1_good_student'

                    # Format timestamp
                    formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    formatted_time = f"{time_part[:2]}-{time_part[2:4]}"

                    # Build folder name
                    if metadata_part:
                        folder_name = os.path.join(base_dir,
                                                   f"session_export_{formatted_date}_{formatted_time}_{metadata_part}")
                    else:
                        folder_name = os.path.join(base_dir, f"session_export_{formatted_date}_{formatted_time}")
                else:
                    # Fallback if format is unexpected
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                    folder_name = os.path.join(base_dir, f"session_export_{timestamp}")
            else:
                # Fallback if session_id doesn't start with 'session_'
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                folder_name = os.path.join(base_dir, f"session_export_{timestamp}")
        else:
            # Original behavior if no session_id provided
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            folder_name = os.path.join(base_dir, f"session_export_{timestamp}")

        os.makedirs(folder_name, exist_ok=True)

        # Define files to save based on mode
        if mode == 1:  # STACER_MODE
            files = {
                "case_study.txt": case_study,
                "cod_stacer.txt": stacer,
                "detailed_feedback.txt": detailed_feedback,
                "interview_transcript.txt": transcript,
                "presentation_transcript.txt": presentation,
            }
        elif mode == 2:  # MI_MODE
            files = {
                "case_study.txt": case_study,
                "MI_EVAL.txt": stacer,
                "detailed_feedback.txt": detailed_feedback,
                "interview_transcript.txt": transcript
            }
        elif mode == 3:  # CBT_MODE
            files = {
                "case_study.txt": case_study,
                "CBT_EVAL.txt": stacer,
                "detailed_feedback.txt": detailed_feedback,
                "interview_transcript.txt": transcript,
                "presentation_transcript.txt": presentation,
            }
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Save each file
        for filename, content in files.items():
            try:
                file_path = os.path.join(folder_name, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logging.info(f"Saved: {filename}")
            except Exception as e:
                logging.error(f"Error saving {filename}: {e}")
                continue

        logging.info(f"Session data saved to: {folder_name}")
        return folder_name


###############################################################
###                   PromptLoader Class                    ###
###############################################################

class PromptLoader:
    """Handles loading of prompts and context files based on mode."""

    def __init__(self, mode: int):
        """
        Initialize PromptLoader with the current mode.

        Args:
            mode: One of STACER_MODE, MI_MODE, or CBT_MODE
        """
        self.mode = mode
        self.file_manager = FileManager(mode)

        # Define prompt paths for each mode
        self.prompt_paths = {
            1: {  # STACER_MODE
                'Clinical Summary': 'Prompts/CBT/clinical_summary_prompt.txt',
                'Virtual Patient': 'Prompts/CBT/virtual_patient_prompt.txt',
                'Verification': 'Prompts/CBT/patient_profile_verification_prompt.txt',
                'Feedback': 'Prompts/CBT/feedback_agent_prompt.txt',
                'CBT Prompt': 'Prompts/CBT/STACER_form_prompt.txt',
                'Virtual Evaluator': 'Prompts/CBT/virtual_evaluator_prompt.txt',
            },
            2: {  # MI_MODE
                'Clinical Summary': 'Prompts/MI/clinical_summary_prompt.txt',
                'Virtual Patient': 'Prompts/MI/virtual_patient_prompt.txt',
                'Verification': 'Prompts/MI/patient_profile_verification_prompt.txt',
                'Feedback': 'Prompts/MI/feedback_agent_prompt.txt',
                'CBT Prompt': 'Prompts/MI/MI_form_prompt.txt',
                'Virtual Evaluator': 'Prompts/MI/virtual_evaluator_prompt.txt',
            },
            3: {  # CBT_MODE
                'Clinical Summary': 'Prompts/CBT/clinical_summary_prompt.txt',
                'Virtual Patient': 'Prompts/CBT/virtual_patient_prompt.txt',
                'Verification': 'Prompts/CBT/patient_profile_verification_prompt.txt',
                'Feedback': 'Prompts/CBT/feedback_agent_prompt.txt',
                'CBT Prompt': 'Prompts/CBT/CBT_form_prompt.txt',
                'Virtual Evaluator': 'Prompts/CBT/virtual_evaluator_prompt.txt',
            }
        }

        # Define context paths for each mode
        self.context_paths = {
            1: {  # STACER_MODE
                'Feedback Criteria': 'Context/CBT/feedback_criteria.txt',
                'Evaluation Form': 'Context/CBT/COD_STACER.txt',
            },
            2: {  # MI_MODE
                'Feedback Criteria': 'Context/MI/feedback_criteria_MI_5point.txt',
                'Evaluation Form': 'Context/MI/MI_EVAL.txt',
            },
            3: {  # CBT_MODE
                'Feedback Criteria': 'Context/CBT/feedback_criteria_CBT.txt',
                'Evaluation Form': 'Context/CBT/CBT_EVAL.txt',
            }
        }

    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt file based on current mode.

        Args:
            prompt_name: Name of the prompt to load (e.g., 'Clinical Summary')

        Returns:
            Prompt content as string

        Raises:
            FileNotFoundError: If mode or prompt name is invalid
        """
        if self.mode not in self.prompt_paths:
            raise FileNotFoundError(f"Invalid mode: {self.mode}")

        if prompt_name not in self.prompt_paths[self.mode]:
            raise FileNotFoundError(
                f"Invalid prompt name for mode {self.mode}: {prompt_name}"
            )

        filepath = self.prompt_paths[self.mode][prompt_name]
        return self.file_manager.read_file(filepath)

    def load_context(self, context_name: str) -> str:
        """
        Load a context file based on current mode.

        Args:
            context_name: Name of the context to load (e.g., 'Feedback Criteria')

        Returns:
            Context content as string

        Raises:
            FileNotFoundError: If mode or context name is invalid
        """
        if self.mode not in self.context_paths:
            raise FileNotFoundError(f"Invalid mode: {self.mode}")

        if context_name not in self.context_paths[self.mode]:
            raise FileNotFoundError(
                f"Invalid context name for mode {self.mode}: {context_name}"
            )

        filepath = self.context_paths[self.mode][context_name]
        return self.file_manager.read_file(filepath)


###############################################################
###                  Formatting Functions                   ###
###############################################################

def format_interview_messages(messages: List) -> Tuple[List, str]:
    """
    Format interview messages for feedback generation.

    Filters out system messages and formats conversation into:
    1. List of LangChain message objects for LLM
    2. Plain text transcript string

    Args:
        messages: List of LangChain message objects (SystemMessage, HumanMessage, AIMessage)

    Returns:
        Tuple of (formatted_langchain_messages, formatted_transcript_string)
    """
    # Filter conversation messages (exclude system messages)
    conversation_messages = [
        m for m in messages
        if (isinstance(m, HumanMessage) or isinstance(m, AIMessage)) and m.content
    ]

    # Format for LangChain (with role prefixes)
    formatted_messages = []
    for m in conversation_messages:
        prefix = "PATIENT: " if isinstance(m, AIMessage) else "PSYCHIATRIST: "
        content = prefix + m.content
        msg_cls = AIMessage if isinstance(m, AIMessage) else HumanMessage
        formatted_messages.append(msg_cls(content=content))

    # Format as plain text transcript
    formatted_lines = []
    for m in conversation_messages:
        prefix = "PATIENT: " if isinstance(m, AIMessage) else "PSYCHIATRIST: "
        formatted_lines.append(prefix + m.content)

    formatted_transcript = "\n".join(formatted_lines)

    return formatted_messages, formatted_transcript


def format_presentation(presentation: List[str]) -> str:
    """
    Format presentation list into a single formatted string.

    Args:
        presentation: List of presentation text entries

    Returns:
        Formatted presentation string
    """
    if not presentation:
        return ""

    formatted_parts = [
        f"Psychiatrist Presentation: {p.strip()}"
        for p in presentation
        if p.strip()
    ]

    return "\n".join(formatted_parts)


def build_langchain_messages(chatlog: List[Dict]) -> List:
    """
    Convert chatlog dictionary format to LangChain message objects.

    Args:
        chatlog: List of message dictionaries with 'type' and 'content' keys

    Returns:
        List of LangChain message objects (SystemMessage, HumanMessage, AIMessage)
    """
    langchain_messages = []

    for msg in chatlog:
        msg_type = msg.get('type')
        content = msg.get('content', '')

        if msg_type == 'system':
            langchain_messages.append(SystemMessage(content=content))
        elif msg_type == 'human':
            langchain_messages.append(HumanMessage(content=content))
        elif msg_type == 'ai':
            langchain_messages.append(AIMessage(content=content))

    return langchain_messages
