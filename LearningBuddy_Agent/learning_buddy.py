import logging
from datetime import datetime
from typing import Optional, Dict
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from utils.api_key_loader import load_openrouter_api_key
from utils.Voice import Voice
from .graph_state import LBState, PatientContext
from .utils import FileManager, PromptLoader
from .workflows import (
    InitializationWorkflow,
    InterviewWorkflow,
    FeedbackWorkflow,
    EvaluatorWorkflow
)

import logging
try:
    import colorlog
except ImportError:
    colorlog = None

# Configure logging
# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear any existing handlers
logger.handlers.clear()

# Create a colored formatter for console when colorlog is installed.
if colorlog is not None:
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
else:
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Create a plain formatter for file output (no colors)
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create and configure file handler
file_handler = logging.FileHandler("learningbuddy.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_formatter)

# Create and configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)



###############################################################
###                  Mode Constants                         ###
###############################################################

# Mode definitions
STACER_MODE = 1
MI_MODE = 2
CBT_MODE = 3


###############################################################
###              Main LearningBuddy Class                   ###
###############################################################

class LearningBuddy:
    """
    Main orchestrator class for LearningBuddy virtual patient simulation.

    Coordinates four main workflows:
    1. Initialization: Setup session and patient profile
    2. Interview: Virtual patient conversation
    3. Feedback: Generate evaluation and CBT form
    4. Evaluator: Q&A about evaluation
    """

    def __init__(self, api_key: str, elevenlabs_api_key: str, mode: int = STACER_MODE):
        """
        Initialize LearningBuddy with API keys and mode.

        Args:
            api_key: OpenAI API key
            elevenlabs_api_key: ElevenLabs API key for voice
            mode: One of STACER_MODE, MI_MODE, or CBT_MODE
        """
        self.api_key = SecretStr(api_key)
        self.elevenlabs_api_key = elevenlabs_api_key
        self.mode = mode
        self.voice = ""
        self.local_agent = True
        self.current_state = None

        # Initialize utilities
        self.file_manager = FileManager(mode)
        self.prompt_loader = PromptLoader(mode)

        # Initialize LLM clients
        self.llm_clients = self._initialize_llm_clients()

        # Initialize workflows
        self.workflows = self._initialize_workflows()

        logging.info(f"LearningBuddy initialized in mode {mode}.")

    def _initialize_llm_clients(self) -> Dict:
        """
        Initialize all LLM clients with appropriate configurations.
        2026-11-09: Adding the option to return the local agent for the verification and summary LLMs.

        Returns:
            Dictionary of LLM client instances
        """

        if self.local_agent:
            def local_llm(temperature=0.2):
                return ChatOpenAI(
                    model="qwen",
                    base_url="http://127.0.0.1:8080/v1",
                    api_key="not-needed",
                    temperature=temperature,
                    max_retries=2,
                    timeout=180
                )
            return {
                "verification": local_llm(0.0),
                "summary": local_llm(0.2),
                "virtual_patient": local_llm(0.7),
                "feedback": local_llm(0.2),
                "virtual_evaluator": local_llm(0.2)
            }
        else:
            return {
                'verification': ChatOpenAI(
                    model="gpt-4.1-mini",
                    temperature=0,
                    api_key=self.api_key,
                    max_retries=5,  # Retry up to 3 times
                    timeout=30
                ),
                'summary': ChatOpenAI(
                    model="gpt-4.1-mini",
                    temperature=0.2,
                    api_key=self.api_key,
                    max_retries=5,  # Retry up to 3 times
                    timeout=30
                ),
                'virtual_patient': ChatOpenAI(
                    model="gpt-4.1-mini",
                    temperature=1.0,
                    api_key=self.api_key,
                    max_retries=5,  # Retry up to 3 times
                    timeout=30,
                    logprobs=True
                ),
                'feedback': ChatOpenAI(
                    model="anthropic/claude-haiku-4.5",
                    temperature=0.2,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=load_openrouter_api_key(),
                    max_retries=5,  # Retry up to 3 times
                    timeout=30
                ),
                'virtual_evaluator': ChatOpenAI(
                    model="google/gemini-2.5-flash-lite-preview-09-2025",
                    temperature=0.2,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=load_openrouter_api_key(),
                    max_retries=5,  # Retry up to 3 times
                    timeout=30
                )
            }

    def _initialize_workflows(self) -> Dict:
        """
        Initialize all workflow instances.

        Returns:
            Dictionary of workflow instances
        """
        return {
            'initialization': InitializationWorkflow(
                llm_verification=self.llm_clients['verification'],
                llm_summary=self.llm_clients['summary'],
                file_manager=self.file_manager,
                prompt_loader=self.prompt_loader
            ),
            'interview': InterviewWorkflow(
                llm_virtual_patient=self.llm_clients['virtual_patient']
            ),
            'feedback': FeedbackWorkflow(
                llm_feedback=self.llm_clients['feedback'],
                llm_virtual_evaluator=self.llm_clients['virtual_evaluator'],
                prompt_loader=self.prompt_loader,
                file_manager=self.file_manager,
                mode=self.mode
            ),
            'evaluator': EvaluatorWorkflow(
                llm_virtual_evaluator=self.llm_clients['virtual_evaluator']
            )
        }

    ###############################################################
    ###                  Public API Methods                     ###
    ###############################################################

    def start_session(
            self,
            case_study: Optional[str] = None,
            selected_profile: Optional[str] = None,
            session_metadata: Optional[dict] = None
    ) -> LBState:
        """
        Start a new session using initialization workflow.

        Args:
            case_study: Pre-loaded case study content (optional)
            selected_profile: Path to specific patient profile (optional)
            session_metadata: Additional metadata for folder naming (optional)
                             Expected keys: 'patient_number', 'student_type'

        Returns:
            Initial state after running initialization workflow

        Raises:
            Exception: If initialization workflow fails
        """
        logging.info("Starting new LearningBuddy session...")

        # Generate session metadata
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')

        # Build session_id with metadata
        session_id_parts = [f"session_{timestamp}"]

        if session_metadata:
            if 'patient_number' in session_metadata:
                session_id_parts.append(f"patient_{session_metadata['patient_number']}")
            if 'student_type' in session_metadata:
                session_id_parts.append(session_metadata['student_type'])

        session_id = "_".join(session_id_parts)

        # Build initial state
        initial_state: LBState = {
            # Conversational state - empty at start
            "interview_messages": [],
            "evaluator_messages": [],

            # Session control
            "current_phase": "initializing",
            "session_active": False,
            "evaluator_active": False,

            # Nested data structures - None at start
            "patient_context": None,
            "presentation": None,
            "evaluation": None,

            # Session metadata
            "metadata": {
                "session_id": session_id,
                "start_time": now.isoformat(),
                "end_time": None,
                "selected_profile": selected_profile,
                "patient_number": session_metadata.get('patient_number') if session_metadata else None,
                "student_type": session_metadata.get('student_type') if session_metadata else None
            },

            # Error handling
            "error": None,

            # Export path
            "export_folder": None
        }

        # If case_study provided, initialize patient_context early
        if case_study:
            initial_state["patient_context"] = PatientContext(
                case_study=case_study,
                diagnosis_summary="",
                clinical_summary="",
                profile_path=selected_profile
            )

        try:
            # Run initialization workflow
            result = self.workflows['initialization'].run(initial_state)
            self.current_state = result

            logging.info(f"Creating Voice")
            self.voice = Voice(self.current_state['patient_context']['case_study'], self.elevenlabs_api_key)

            # Log success or errors
            if result.get("error"):
                logging.error(f"Session initialized with errors: {result['error']}")
            else:
                logging.info(f"Session initialized successfully. ID: {session_id}")

            return result

        except Exception as e:
            logging.error(f"Error starting session: {str(e)}", exc_info=True)

            # Return error state
            error_state = initial_state
            error_state["current_phase"] = "error"
            error_state["error"] = f"Initialization failed: {str(e)}"
            return error_state

    def send_message(self, user_input: str) -> Dict:
        """
        Send a user message to the virtual patient.

        Args:
            user_input: User's message/question to the virtual patient

        Returns:
            Dictionary with AI response and session status
        """
        if not self.current_state or not self.current_state.get("session_active"):
            logging.error("No active session found.")
            return {"error": "No active session"}

        try:
            # Run interview workflow
            result = self.workflows['interview'].run(self.current_state, user_input)
            self.current_state = result

            # Extract AI response
            interview_messages = result.get("interview_messages", [])
            ai_response = interview_messages[-1].content if interview_messages else ""

            return {
                "ai_response": ai_response,
                "session_active": result.get("session_active", True),
                "current_phase": "active_roleplay",
                "interview_messages": interview_messages
            }

        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            return {"error": str(e)}

    def add_presentation_message(self, user_input: str) -> Dict:
        """
        Add a message to the case presentation.

        Args:
            user_input: User's presentation text

        Returns:
            Dictionary with updated presentation and status
        """
        if not self.current_state:
            logging.error("No active session found.")
            return {"error": "No active session"}

        try:
            # Get or initialize presentation list
            presentation = self.current_state.get("presentation", [])
            if presentation is None:
                presentation = []

            # Add new presentation message
            presentation.append(user_input)

            # Update state
            self.current_state["presentation"] = presentation

            logging.info(f"Presentation input added: {user_input[:50]}...")

            return {
                "presentation": presentation,
                "status": "updated"
            }

        except Exception as e:
            logging.error(f"Error adding presentation message: {str(e)}")
            return {"error": str(e)}

    def end_session(self) -> LBState:
        """
        End the session and generate feedback.

        Returns:
            Final state with feedback and saved data
        """
        if not self.current_state:
            logging.error("No active session to end.")
            return {"error": "No active session"}

        try:
            # Run feedback workflow
            result = self.workflows['feedback'].run(self.current_state)
            self.current_state = result

            logging.info("Session ended successfully.")
            return result

        except Exception as e:
            logging.error(f"Error ending session: {str(e)}")
            return {"error": str(e)}

    def send_evaluator_message(self, user_input: str) -> Dict:
        """
        Send a message to the virtual evaluator for Q&A about feedback.

        Args:
            user_input: User's question about the evaluation

        Returns:
            Dictionary with evaluator response
        """
        if not self.current_state or not self.current_state.get("evaluator_active"):
            logging.error("No active evaluation session.")
            return {"error": "No active evaluation session"}

        try:
            # Run evaluator workflow
            result = self.workflows['evaluator'].run(self.current_state, user_input)
            self.current_state = result

            # Extract AI response
            evaluator_messages = result.get("evaluator_messages", [])
            ai_response = evaluator_messages[-1].content if evaluator_messages else ""

            return {
                "ai_response": ai_response,
                "evaluator_active": result.get("evaluator_active", True),
                "current_phase": "evaluator_chat",
                "evaluator_messages": evaluator_messages
            }

        except Exception as e:
            logging.error(f"Error in evaluator chat: {str(e)}")
            return {"error": str(e)}

    ###############################################################
    ###              Session Management Methods                 ###
    ###############################################################

    def get_session_state(self) -> Optional[LBState]:
        """
        Get the current session state.

        Returns:
            Current state or None if no active session
        """
        return self.current_state

    def reset_session(self):
        """Reset current session state."""
        self.current_state = None
        logging.info("Session state reset.")

    def get_session_statistics(self) -> Dict:
        """
        Get current session statistics.

        Returns:
            Dictionary with session statistics
        """
        if not self.current_state:
            return {}

        interview_messages = self.current_state.get("interview_messages", [])
        from langchain_core.messages import AIMessage, HumanMessage

        conversation_messages = [
            m for m in interview_messages
            if isinstance(m, (HumanMessage, AIMessage))
        ]

        return {
            "total_messages": len(conversation_messages),
            "total_exchanges": len(conversation_messages) // 2,
            "current_phase": self.current_state.get("current_phase"),
            "session_active": self.current_state.get("session_active", False),
            "metadata": self.current_state.get("metadata", {})
        }

    def export_session_data(self) -> Optional[Dict]:
        """
        Export session data for analysis.

        Returns:
            Dictionary with all session data or None if no session
        """
        if not self.current_state:
            return None

        patient_ctx = self.current_state.get("patient_context", {})

        return {
            "metadata": self.current_state.get("metadata", {}),
            "case_study": patient_ctx.get("case_study") if patient_ctx else None,
            "clinical_summary": patient_ctx.get("clinical_summary") if patient_ctx else None,
            "diagnosis_summary": patient_ctx.get("diagnosis_summary") if patient_ctx else None,
            "interview_messages": self.current_state.get("interview_messages", []),
            "presentation": self.current_state.get("presentation", []),
            "cod_stacer": self.current_state.get("cod_stacer"),
            "formatted_transcript": self.current_state.get("formatted_transcript"),
            "formatted_presentation": self.current_state.get("formatted_presentation"),
            "export_folder": self.current_state.get("export_folder"),
            "session_statistics": self.get_session_statistics()
        }

    ###############################################################
    ###              Utility Methods (Backward Compatibility)   ###
    ###############################################################

    def load_patient_profile(self, filepath: str) -> str:
        """
        Load a specific patient profile.
        Wrapper for FileManager method for backward compatibility.

        Args:
            filepath: Path to patient profile file

        Returns:
            Patient profile content
        """
        return self.file_manager.load_patient_profile(filepath)
