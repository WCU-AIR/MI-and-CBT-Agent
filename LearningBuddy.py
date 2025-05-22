
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import SecretStr
from FileReader import load_prompt, load_context, load_patient_profile, load_random_case_profile

#Definition of graph i/o types
class InterviewState(TypedDict):
    case_study: Optional[str]
    diagnosis_summary: str
    clinical_summary: str
    chatlog: List[Dict[str, Any]]
    feedback: str

class LearningBuddy:
    def __init__(self, api_key: str):
        self.api_key = SecretStr(api_key)
        self.llm_mini = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=self.api_key)
        self.llm_mini_virtualPatient = ChatOpenAI(model="gpt-4.1-mini", temperature=1, api_key=self.api_key)
        self.graph = self._build_graph()

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

        return workflow.compile()

    def run(self, case_study: Optional[str] = None):
        result = self.graph.invoke({"case_study": case_study})
        print(result["feedback"])
        return result

    # ========== Workflow Phases ==========

    def _phase_1_profile_verification(self, state):
        case_study = state["case_study"]
        if case_study is None:
            case_study = load_random_case_profile()

        max_attempts = 10
        attempt = 0
        while True:
            attempt += 1
            if attempt > max_attempts:
                raise RuntimeError("No valid case study found after multiple attempts.")

            if case_study is None:
                case_study = load_random_case_profile()

            instruction_prompt = load_prompt('Verification')
            messages = [SystemMessage(content=f"{instruction_prompt}\n\n{case_study}")]
            response = self.llm_mini.invoke(messages)
            response_text = response.content.strip()

            if response_text.lower().startswith("no"):
                print(f"[Attempt {attempt}] Case study rejected by LLM.\nFull response:\n{response_text}\n")
                case_study = None
                continue
            else:
                break

        return {
            "case_study": case_study,
            "diagnosis_summary": response.content
        }

    def _phase_2_summary(self, state):
        instruction_prompt = load_prompt('Clinical Summary')
        case_study = state["case_study"]

        messages = [
            SystemMessage(content=instruction_prompt),
            HumanMessage(content=case_study)
        ]

        response = self.llm_mini.invoke(messages)
        assistant_message = response.content

        print("\n(Clinical Summary):")
        for sentence in assistant_message.split('. '):
            if sentence.strip():
                print(sentence.strip() + ('.' if not sentence.endswith('.') else ''))

        return {"clinical_summary": assistant_message}

    def _phase_3_roleplay(self, state):
        instruction_prompt = load_prompt('Virtual Patient')
        case_study = state["case_study"]

        messages = [
            SystemMessage(content=instruction_prompt),
            HumanMessage(content=case_study)
        ]

        while True:
            user_input = input("\nYou (Therapist): ")
            if user_input.lower() in {"exit", "quit"}:
                print("Session ended.")
                break

            messages.append(HumanMessage(content=user_input))
            response = self.llm_mini_virtualPatient.invoke(messages)
            assistant_message = response.content
            messages.append(AIMessage(content=assistant_message))

            print("\n(Patient):")
            for sentence in assistant_message.split('. '):
                if sentence.strip():
                    print(sentence.strip() + ('.' if not sentence.endswith('.') else ''))

        return {"chatlog": [m.model_dump() for m in messages]}

    def _phase_4_feedback(self, state):
        evaluation_context = load_context('Feedback Criteria')
        instruction_prompt = load_prompt('Feedback')
        chatlog = state["chatlog"]

        normalized_chatlog = []
        for m in chatlog:
            role = m.get("role") or ("user" if m.get("type") == "human" else "assistant" if m.get("type") == "ai" else m.get("type"))
            if role:
                normalized_chatlog.append({**m, "role": role})

        messages = [
            SystemMessage(content=f"{instruction_prompt}\n\n{evaluation_context}"),
            HumanMessage(content="Here is the chatlog of the interview. Please provide detailed feedback."),
        ] + [
            SystemMessage(**m) if m["role"] == "system" else
            HumanMessage(**m) if m["role"] == "user" else
            AIMessage(**m)
            for m in normalized_chatlog
        ]

        response = self.llm_mini.invoke(messages)
        return {"feedback": response.content}
