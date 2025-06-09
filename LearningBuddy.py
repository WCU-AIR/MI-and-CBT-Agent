import logging
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import SecretStr
import random
from datetime import datetime
from Voice import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("learningbuddy.log"),
        logging.StreamHandler()
    ]
)

# Defining LangGraph State Variables
class LBState(TypedDict):
    case_study: Optional[str]
    diagnosis_summary: str
    clinical_summary: str
    chatlog: List[Dict[str, Any]]
    feedback: str
    current_phase: str
    session_active: bool
    user_input: Optional[str]
    ai_response: Optional[str]
    session_metadata: Dict[str, Any]
    error_message: Optional[str]
    next_action: Optional[str]
    cod_stacer: str

class LearningBuddy:
    def __init__(self, api_key: str):
        self.api_key = SecretStr(api_key)
        self.llm_mini_verification = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=self.api_key)
        self.llm_mini_summary = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2, api_key=self.api_key)
        self.llm_mini_virtualPatient = ChatOpenAI(model="gpt-4.1-mini", temperature=1, api_key=self.api_key)
        self.llm_feedback = ChatOpenAI(model="gpt-4.1", temperature=0.1, api_key=self.api_key)
        self.init_graph = self._build_init_workflow()
        self.message_graph = self._build_message_workflow()
        self.feedback_graph = self._build_feedback_workflow()
        self.voice = ""
        self.current_state = None
        logging.info("LearningBuddy initialized with separate LangGraph workflows.")

    ###############################################################
    ###              Workflow 1: Initialization                 ###
    ###############################################################
    #   This workflow initializes the virtual patient interview:
    #            1. Selects a patient profile
    #            2. Verifies the selected profile
    #            3. Generates an introductory 'clinical summary'
    #            4. Creates the prompt to be used for the chatbot
    #
    #   Public function to use this workflow: start_session()

    def _build_init_workflow(self):
        """Build initialization workflow - linear, no loops"""
        workflow = StateGraph(state_schema=LBState)
        
        workflow.add_node("initialize", RunnableLambda(self._select_patient_profile))
        workflow.add_node("verify_profile", RunnableLambda(self._profile_verification))
        workflow.add_node("generate_summary", RunnableLambda(self._clinical_summary))
        workflow.add_node("setup_roleplay", RunnableLambda(self._roleplay_setup))
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "verify_profile")
        workflow.add_edge("verify_profile", "generate_summary")
        workflow.add_edge("generate_summary", "setup_roleplay")
        workflow.add_edge("setup_roleplay", END)

        logging.info("Initialization workflow built successfully.")
        return workflow.compile()

    def _select_patient_profile(self, state: LBState):
        """ Checks if user selected a specific patient profile.
            Uses the selected profile if it exists,
            Else, chooses a random profile.
        """
        logging.info("Initializing session...")

        case_study = state.get("case_study")
        selected_profile = state.get("session_metadata", {}).get("selected_profile")

        # Load case study if not provided
        if not case_study:
            try:
                if selected_profile:
                    case_study = self.load_patient_profile(selected_profile)
                    logging.info(f"Loaded selected profile: {selected_profile}")
                else:
                    case_study = self._load_random_case_profile()
                    logging.info("Loaded random profile")
            except Exception as e:
                logging.error(f"Error loading case study: {str(e)}")
                return {**state, "error_message": f"Failed to load case study: {str(e)}"}

        return {
            **state,
            "case_study": case_study,
            "current_phase": "verify_profile"
        }

    def _profile_verification(self, state: LBState):
        """ Verifies that selected profile is a patient with MDD.
            If profile doesn't have MDD, chooses a different random profile.
        """
        logging.info("Starting profile verification phase.")
        case_study = state["case_study"]

        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            try:
                instruction_prompt = self._load_prompt('Verification')
                messages = [SystemMessage(content=f"{instruction_prompt}\n\n{case_study}")]
                response = self.llm_mini_verification.invoke(messages)
                response_text = response.content.strip()

                if response_text.lower().startswith("no"):
                    logging.info(f"[Attempt {attempt}] Case study rejected by LLM.")
                    if attempt >= max_attempts:
                        # If all attempts fail, just proceed with the current case study
                        logging.warning("Max verification attempts reached, proceeding with current case study.")
                        response_text = "Yes, proceeding with case study."
                        break
                    # Try loading a new random profile
                    case_study = self._load_random_case_profile()
                    continue
                else:
                    logging.info("Case study verified successfully.")
                    break
            except Exception as e:
                logging.error(f"Error in profile verification: {str(e)}")
                return {**state, "error_message": f"Profile verification failed: {str(e)}"}

        self.voice = set_patient_voice(case_study)

        return {
            **state,
            "case_study": case_study,
            "diagnosis_summary": response.content,
            "current_phase": "generate_summary"
        }

    def _clinical_summary(self, state: LBState):
        """ Generates a clinical summary,
            This summary includes some brief intake information
            on the patient before the interview starts
        """
        logging.info("Starting clinical summary phase.")

        try:
            instruction_prompt = self._load_prompt('Clinical Summary')
            case_study = state["case_study"]

            messages = [
                SystemMessage(content=instruction_prompt),
                HumanMessage(content=case_study)
            ]

            response = self.llm_mini_summary.invoke(messages)
            assistant_message = response.content
            logging.info("Clinical summary generated.")

            return {
                **state,
                "clinical_summary": assistant_message,
                "current_phase": "setup_roleplay"
            }
        except Exception as e:
            logging.error(f"Error in clinical summary: {str(e)}")
            return {**state, "error_message": f"Clinical summary failed: {str(e)}"}

    def _roleplay_setup(self, state: LBState):
        """ Setup roleplay and complete initialization.
            Loads and formats the prompt to be used
            for the virtual patient chatbot.
        """
        logging.info("Setting up virtual patient roleplay.")

        try:
            instruction_prompt = self._load_prompt('Virtual Patient')
            case_study = state["case_study"]

            # Initialize chatlog with system and case study messages
            initial_messages = [
                {
                    "type": "system",
                    "role": "system",
                    "content": instruction_prompt
                },
                {
                    "type": "human",
                    "role": "user",
                    "content": case_study
                }
            ]

            logging.info("Roleplay setup completed. Session ready for interaction.")
            return {
                **state,
                "chatlog": initial_messages,
                "current_phase": "active_roleplay",
                "session_active": True,
                "user_input": None,
                "ai_response": None
            }
        except Exception as e:
            logging.error(f"Error in roleplay setup: {str(e)}")
            return {**state, "error_message": f"Roleplay setup failed: {str(e)}"}

    def start_session(self, case_study: Optional[str] = None, selected_profile: Optional[str] = None):
        """ Start a new session using initialization workflow.
            Public function for use by frontend to use this workflow.
        """
        logging.info("Starting new LearningBuddy session...")

        initial_state = {
            "case_study": case_study,
            "diagnosis_summary": "",
            "clinical_summary": "",
            "chatlog": [],
            "feedback": "",
            "cod_stacer": "",
            "current_phase": "initialize",
            "session_active": False,
            "user_input": None,
            "ai_response": None,
            "session_metadata": {
                "start_time": datetime.now().isoformat(),
                "selected_profile": selected_profile,
                "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "error_message": None,
            "next_action": None
        }

        try:
            # Run initialization workflow
            result = self.init_graph.invoke(initial_state)
            self.current_state = result
            logging.info("Session initialized successfully.")
            return result
        except Exception as e:
            logging.error(f"Error starting session: {str(e)}")
            return {"error_message": str(e), "current_phase": "error"}


    ###############################################################
    ###           Workflow 2: Virtual Patient Chatbot           ###
    ###############################################################
    #   This workflow handles one turn of the virtual patient interview:
    #            1. Read user input
    #            2. Generate patient response
    #
    #   Public function to use this workflow: send_message()

    def _build_message_workflow(self):
        """ Build message processing workflow - linear, no loops"""
        workflow = StateGraph(state_schema=LBState)
        
        workflow.add_node("process_input", RunnableLambda(self._process_user_input))
        workflow.add_node("generate_response", RunnableLambda(self._generate_ai_response))
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "generate_response")
        workflow.add_edge("generate_response", END)

        logging.info("Message processing workflow built successfully.")
        return workflow.compile()

    def _process_user_input(self, state: LBState):
        """ Process user input during interview.
            Appends all entries to chatlog for future reference.
        """
        user_input = state.get("user_input")
        if not user_input:
            return state

        logging.info(f"Processing user input: {user_input[:50]}...")

        # Add user message to chatlog
        chatlog = state.get("chatlog", [])
        chatlog.append({
            "type": "human",
            "role": "user",
            "content": user_input
        })

        return {
            **state,
            "chatlog": chatlog,
            "current_phase": "generate_response",
            "user_input": None
        }

    def _generate_ai_response(self, state: LBState):
        """ Generate AI response during roleplay.
            Appends all entries to chatlog for future reference.
        """
        logging.info("Generating AI patient response...")

        try:
            chatlog = state.get("chatlog", [])

            langchain_messages = []
            for msg in chatlog:
                if msg.get('type') == 'system':
                    langchain_messages.append(SystemMessage(content=msg['content']))
                elif msg.get('type') == 'human':
                    langchain_messages.append(HumanMessage(content=msg['content']))
                elif msg.get('type') == 'ai':
                    langchain_messages.append(AIMessage(content=msg['content']))

            # Generate response
            response = self.llm_mini_virtualPatient.invoke(langchain_messages)
            ai_response = response.content

            # Add AI response to chatlog
            chatlog.append({
                "type": "ai",
                "role": "assistant",
                "content": ai_response
            })

            logging.info("AI response generated successfully.")

            return {
                **state,
                "chatlog": chatlog,
                "ai_response": ai_response,
                "session_active": True,
                "current_phase": "message_complete"
            }
        except Exception as e:
            logging.error(f"Error generating AI response: {str(e)}")
            return {**state, "error_message": f"AI response generation failed: {str(e)}"}

    def send_message(self, user_input: str):
        """ Send a user message using message processing workflow.
            Public function for use by frontend to use this workflow.
        """
        if not self.current_state or not self.current_state.get("session_active"):
            logging.error("No active session found.")
            return {"error": "No active session"}

        try:
            # Create state for message processing
            message_state = {**self.current_state, "user_input": user_input}
            result = self.message_graph.invoke(message_state)
            self.current_state = result

            return {
                "ai_response": result.get("ai_response"),
                "session_active": result.get("session_active", True),
                "current_phase": "active_roleplay",
                "chatlog": result.get("chatlog", [])
            }
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            return {"error": str(e)}

    ###############################################################
    ###                Workflow 3: STACER Feedback              ###
    ###############################################################
    #   This workflow generates all feedback after an interview ends:
    #            1. Generates detailed feedback and fills out STACER
    #            2. Ends current session and saves all session data
    #
    #   Public function to use this workflow: end_session()

    def _build_feedback_workflow(self):
        """Build feedback workflow - linear, no loops"""
        workflow = StateGraph(state_schema=LBState)
        
        workflow.add_node("evaluate_session", RunnableLambda(self._generate_feedback))
        workflow.add_node("complete_session", RunnableLambda(self._complete_session))
        workflow.set_entry_point("evaluate_session")
        workflow.add_edge("evaluate_session", "complete_session")
        workflow.add_edge("complete_session", END)

        logging.info("Feedback workflow built successfully.")
        return workflow.compile()

    def _generate_feedback(self, state: LBState):
        """ Generate comprehensive feedback.
            First scores all defined criteria on a 1-5 scale and writes
            thorough feedback based on provided rubric.
            Then uses full feedback to fill out a formatted COD STACER.
        """
        logging.info("Starting feedback evaluation phase.")

        # Validate and format required data
        case_study = state.get("case_study")
        clinical_summary = state.get("clinical_summary")
        chatlog = state.get("chatlog", [])

        if not case_study:
            return {**state, "error_message": "Case study content is missing from state."}
        if not clinical_summary:
            return {**state, "error_message": "Clinical summary is missing from state."}
        if not chatlog:
            return {**state, "error_message": "Chatlog is empty or missing from state."}

        try:
            evaluation_context = self._load_context('Feedback Criteria') or ""
            base_instruction_prompt = self._load_prompt('Feedback') or ""
        except Exception as e:
            logging.exception("Failed to load feedback prompt/context")
            return {**state, "error_message": f"Error loading prompts or context: {str(e)}"}

        strict_feedback_prompt = f"""
                                {base_instruction_prompt}

                                {evaluation_context}

                                Patient Profile for Reference:
                                {case_study}

                                Clinical Summary:
                                {clinical_summary}
                                """
        # Filter conversation messages (exclude system messages and case study repeats)
        conversation_messages = [
            m for m in chatlog
            if m.get("role") in ["user", "assistant"] and
               m.get("content") and
               m["content"].strip() != case_study.strip()
        ]

        messages = [
            SystemMessage(content=strict_feedback_prompt),
            HumanMessage(content="Here is the therapy session transcript to evaluate:")
        ]

        formatted_lines = []
        for m in conversation_messages:
            prefix = "THERAPIST: " if m["role"] == "user" else "PATIENT: "
            content = prefix + m["content"]
            msg_cls = HumanMessage if m["role"] == "user" else AIMessage
            messages.append(msg_cls(content=content))
            formatted_lines.append(content)

        formatted_transcript = "\n".join(formatted_lines)

        # Generate Full Feedback with scores
        messages.append(HumanMessage(content="""
                                            Now provide your comprehensive, strict evaluation of this therapy session. 
                                            Be thorough, critical, and specific in identifying areas for improvement.
                                            Structure your feedback using all the sections mentioned.
                                            """))

        try:
            response = self.llm_feedback.invoke(messages)
            feedback_text = response.content.strip()
        except Exception as e:
            logging.exception("LLM feedback generation failed")
            return {**state, "error_message": f"Failed to generate feedback: {str(e)}"}

        num_exchanges = len(conversation_messages) // 2
        session_feedback = f"""
                            ## SESSION STATISTICS
                            - Total Exchanges: {num_exchanges}
                            - Session Completeness: {'Incomplete - ended prematurely' if num_exchanges < 5 else 'Complete'}

                            ## DETAILED CLINICAL EVALUATION

                            {feedback_text}

                            ---
                            *Note: This feedback is intentionally strict and critical to accelerate your learning.*
                            """

        # Generate Formatted COD STACER Form
        instruction_prompt = self._load_prompt('STACER Prompt')
        blank_stacer_file = self._load_context("COD STACER")

        stacer_prompt = f"""
        {instruction_prompt}

        Here is the Full Transcript:
        {formatted_transcript}

        Here is the Evaluation Criteria to base your answer off of:
        {feedback_text}

        And here is the blank STACER form that you must use to format your answer:
        {blank_stacer_file}
        """

        messages = [
            SystemMessage(content=stacer_prompt),
            HumanMessage(content="Please provide the formatted answer below:")
        ]

        response = self.llm_feedback.invoke(messages)
        cod_stacer_response = response.content

        logging.info("Feedback evaluation complete.")
        return {
            **state,
            "feedback": session_feedback,
            'cod_stacer': cod_stacer_response,
            "current_phase": "complete_session",
            "session_active": False
        }

    def _complete_session(self, state: LBState):
        """Complete the session by saving all session data"""
        logging.info("Session completed successfully.")

        # Update session metadata
        session_metadata = state.get("session_metadata", {})
        session_metadata.update({
            "end_time": datetime.now().isoformat(),
            "total_exchanges": len(
                [m for m in state.get("chatlog", []) if m.get("role") in ["user", "assistant"]]) // 2,
            "session_duration_minutes": self._calculate_session_duration(session_metadata.get("start_time")),
            "completion_status": "completed"
        })

        return {
            **state,
            "session_metadata": session_metadata,
            "current_phase": "completed",
            "session_active": False
        }

    def end_session(self):
        """ End session using feedback workflow
            Public function for use by frontend to use this workflow.
        """
        if not self.current_state:
            return {"error": "No active session"}
        
        try:
            result = self.feedback_graph.invoke(self.current_state)
            self.current_state = result
            
            return result
        except Exception as e:
            logging.error(f"Error ending session: {str(e)}")
            return {"error": str(e)}


###############################################################
###                Additional Utility Methods               ###
###############################################################

    def get_session_state(self):
        """Get current session state"""
        return self.current_state

    def reset_session(self):
        """Reset current session state"""
        self.current_state = None
        logging.info("Session state reset.")

    def get_session_statistics(self):
        """Get current session statistics"""
        if not self.current_state:
            return {}
        
        chatlog = self.current_state.get("chatlog", [])
        conversation_messages = [m for m in chatlog if m.get("role") in ["user", "assistant"]]
        
        return {
            "total_messages": len(conversation_messages),
            "total_exchanges": len(conversation_messages) // 2,
            "current_phase": self.current_state.get("current_phase"),
            "session_active": self.current_state.get("session_active", False),
            "session_metadata": self.current_state.get("session_metadata", {})
        }

    def export_session_data(self):
        """Export session data for analysis"""
        if not self.current_state:
            return None
        
        return {
            "session_metadata": self.current_state.get("session_metadata", {}),
            "case_study": self.current_state.get("case_study"),
            "clinical_summary": self.current_state.get("clinical_summary"),
            "diagnosis_summary": self.current_state.get("diagnosis_summary"),
            "chatlog": self.current_state.get("chatlog", []),
            "feedback": self.current_state.get("feedback"),
            "session_statistics": self.get_session_statistics(),
            "cod_stacer": self.current_state.get("cod_stacer"),
        }

    def _calculate_session_duration(self, start_time_iso: str) -> float:
        """Calculate session duration in minutes"""
        try:
            start_time = datetime.fromisoformat(start_time_iso)
            duration = datetime.now() - start_time
            return duration.total_seconds() / 60
        except:
            return 0.0

    def _read_file(self, filepath):
        """ Reads and returns the selected file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Case study file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to read case study: {e}")

    def load_patient_profile(self, filepath: str) -> str:
        """ Returns the selected patient profile"""
        return self._read_file(filepath)

    def _load_random_case_profile(self, directory='Patient_Profiles'):
        """ Returns a randomly selected patient profile"""
        files = [f for f in os.listdir(directory)]
        if not files:
            raise FileNotFoundError("No patient profiles found in the directory.")
        random_file = random.choice(files)
        full_path = os.path.join(directory, random_file)
        return self.load_patient_profile(full_path)

    def _load_prompt(self, prompt_name: str) -> str:
        """ Returns the selected prompt"""
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
        return self._read_file(filepath)

    def _load_context(self, context_name: str) -> str:
        """ Returns the selected LLM context"""
        if context_name == 'Feedback Criteria':
            filepath = 'Context/feedback_criteria.txt'
        elif context_name == 'COD STACER':
            filepath = 'Context/COD_STACER.txt'
        else:
            filepath = ''
        return self._read_file(filepath)