import logging
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import SecretStr
from FileReader import load_prompt, load_context, load_patient_profile, load_random_case_profile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("learningbuddy.log"),
        logging.StreamHandler()
    ]
)

class InterviewState(TypedDict):
    case_study: Optional[str]
    diagnosis_summary: str
    clinical_summary: str
    chatlog: List[Dict[str, Any]]
    feedback: str

class LearningBuddy:
    def __init__(self, api_key: str):
        self.api_key = SecretStr(api_key)
        self.llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)
        self.llm_mini_virtualPatient = ChatOpenAI(model="gpt-4o-mini", temperature=1, api_key=self.api_key)
        self.llm_feedback = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=self.api_key)
        self.graph = self._build_graph()
        logging.info("LearningBuddy initialized.")

    def _build_graph(self):
        workflow = StateGraph(state_schema=InterviewState)
        workflow.add_node("profile-verification", RunnableLambda(self._phase_1_profile_verification))
        workflow.add_node("clinical-summary", RunnableLambda(self._phase_2_summary))
        workflow.add_node("roleplay", RunnableLambda(self._phase_3_roleplay))
        workflow.add_node("evaluate", RunnableLambda(self._phase_4_feedback))

        workflow.set_entry_point("profile-verification")
        workflow.add_edge("profile-verification", "clinical-summary")
        workflow.add_edge("clinical-summary", "roleplay")
        workflow.add_edge("roleplay", "evaluate")
        workflow.set_finish_point("evaluate")

        logging.info("Graph workflow built successfully.")
        return workflow.compile()

    def run(self, case_study: Optional[str] = None):
        logging.info("Running LearningBuddy workflow...")
        result = self.graph.invoke({"case_study": case_study})
        logging.info("Workflow complete. Returning feedback.")
        return result

    def _phase_1_profile_verification(self, state):
        logging.info("Starting profile verification phase.")
        case_study = state["case_study"]
        if case_study is None:
            case_study = load_random_case_profile()

        max_attempts = 10
        attempt = 0
        while True:
            attempt += 1
            if attempt > max_attempts:
                logging.error("No valid case study found after multiple attempts.")
                raise RuntimeError("No valid case study found after multiple attempts.")

            if case_study is None:
                case_study = load_random_case_profile()

            instruction_prompt = load_prompt('Verification')
            messages = [SystemMessage(content=f"{instruction_prompt}\n\n{case_study}")]
            response = self.llm_mini.invoke(messages)
            response_text = response.content.strip()

            if response_text.lower().startswith("no"):
                logging.info(f"[Attempt {attempt}] Case study rejected by LLM.\nFull response:\n{response_text}")
                case_study = None
                continue
            else:
                break

        return {
            **state,
            "case_study": case_study,
            "diagnosis_summary": response.content
        }

    def _phase_2_summary(self, state):
        logging.info("Starting clinical summary phase.")
        instruction_prompt = load_prompt('Clinical Summary')
        case_study = state["case_study"]

        messages = [
            SystemMessage(content=instruction_prompt),
            HumanMessage(content=case_study)
        ]

        response = self.llm_mini.invoke(messages)
        assistant_message = response.content
        logging.info("Clinical summary generated.")

        return {
            **state,
            "clinical_summary": assistant_message
        }

    def _phase_3_roleplay(self, state):
        logging.info("Starting virtual patient roleplay phase.")
        instruction_prompt = load_prompt('Virtual Patient')
        case_study = state["case_study"]

        messages = [
            SystemMessage(content=instruction_prompt),
            HumanMessage(content=case_study)
        ]

        while True:
            user_input = input("\nYou (Therapist): ")
            if user_input.lower() in {"exit", "quit"}:
                logging.info("Session ended by user.")
                break

            messages.append(HumanMessage(content=user_input))
            response = self.llm_mini_virtualPatient.invoke(messages)
            assistant_message = response.content
            messages.append(AIMessage(content=assistant_message))

            print("\n(Patient):")
            print(assistant_message)

        return {
            **state,
            "chatlog": [m.model_dump() for m in messages]
        }




    def _phase_4_feedback(self, state):
        logging.info("Starting feedback evaluation phase.")

        case_study = state.get("case_study")
        clinical_summary = state.get("clinical_summary")
        chatlog = state.get("chatlog")

        # Ensure required data exists
        if not case_study:
            raise ValueError("Case study content is missing from state.")
        if not clinical_summary:
            raise ValueError("Clinical summary is missing from state.")
        if not chatlog:
            raise ValueError("Chatlog is empty or missing from state.")

        try:
            evaluation_context = load_context('Feedback Criteria') or ""
            base_instruction_prompt = load_prompt('Feedback') or ""
        except Exception as e:
            logging.exception("Failed to load feedback prompt/context")
            raise RuntimeError(f"Error loading prompts or context: {e}")

        strict_feedback_prompt = f"""
    {base_instruction_prompt}

    {evaluation_context}

    Patient Profile for Reference:
    {case_study}

    Clinical Summary:
    {clinical_summary}

    """

        # Normalize chat messages
        normalized_chatlog = []
        for m in chatlog:
            role = m.get("role") or m.get("type")
            if role:
                normalized_chatlog.append({**m, "role": role})

        # Filter out system messages and case study repeats
        conversation_messages = [
            m for m in normalized_chatlog
            if m["role"] in ["user", "assistant"] and m.get("content") and m["content"].strip() != case_study.strip()
        ]

        # Build messages for the feedback LLM
        messages = [
            SystemMessage(content=strict_feedback_prompt),
            HumanMessage(content="Here is the therapy session transcript to evaluate:")
        ]

        for m in conversation_messages:
            prefix = "THERAPIST: " if m["role"] == "user" else "PATIENT: "
            content = prefix + m["content"]
            msg_cls = HumanMessage if m["role"] == "user" else AIMessage
            messages.append(msg_cls(content=content))

        messages.append(HumanMessage(content="""
    Now provide your comprehensive, strict evaluation of this therapy session. 
    Be thorough, critical, and specific in identifying areas for improvement.
    Structure your feedback using all the sections mentioned.
    """))

        # Call LLM to generate feedback
        try:
            response = self.llm_feedback.invoke(messages)
            feedback_text = response.content.strip()
        except Exception as e:
            logging.exception("LLM feedback generation failed")
            raise RuntimeError(f"Failed to generate feedback: {e}")

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

        logging.info("Feedback evaluation complete.")
        return {
            **state,
            "feedback": session_feedback
        }



    def _analyze_conversation_quality(self, messages: List[Dict]):
        therapist_messages = [m for m in messages if m.get("role") == "user"]

        analysis = {
            "open_questions": 0,
            "closed_questions": 0,
            "empathetic_responses": 0,
            "missed_cues": 0,
            "boundary_issues": 0,
            "therapeutic_techniques": []
        }

        for msg in therapist_messages:
            content = msg.get("content", "").lower()

            if "?" in content:
                if any(word in content for word in ["how", "what", "tell me about", "describe", "explain"]):
                    analysis["open_questions"] += 1
                elif any(word in content for word in ["do you", "are you", "have you", "is it"]):
                    analysis["closed_questions"] += 1

            if any(phrase in content for phrase in ["i understand", "that must be", "i hear you", "it sounds like"]):
                analysis["empathetic_responses"] += 1

        return analysis
