
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import TypedDict, List, Dict, Any
from pydantic import SecretStr
from FileReader import load_prompt, load_context, load_patient_profile

# Initialize LLMs
apiKey = SecretStr('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')
llm_mini = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=apiKey)
llm_mini_virtualPatient = ChatOpenAI(model="gpt-4.1-mini", temperature=1, api_key=apiKey)
#llm_nano = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0, openai_api_key=apiKey)


# Phase 1: Profile Verification
def phase_1_profile_verification(state):
    case_study = state["case_study"]
    instruction_prompt = load_prompt('Verification')

    messages = [
        SystemMessage(content=f"""{instruction_prompt}\n\n{case_study}""")
    ]

    response = llm_mini.invoke(messages)
    return {"diagnosis_summary": response.content}

# Phase 2: Generate Clinical Summary
def phase_2_summary(state):
    instruction_prompt = load_prompt('Clinical Summary')
    case_study = state["case_study"]

    messages = [
        SystemMessage(content=instruction_prompt),
        HumanMessage(content=case_study)
    ]

    response = llm_mini.invoke(messages)
    assistant_message = response.content

    print("\n(Clinical Summary):")
    for sentence in assistant_message.split('. '):
        if sentence.strip():
            print(sentence.strip() + ('.' if not sentence.endswith('.') else ''))

    return {"clinical_summary": assistant_message}

# Phase 3: Roleplay with virtual patient (interactive loop)
def phase_3_roleplay(state):
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
        response = llm_mini_virtualPatient.invoke(messages)
        assistant_message = response.content
        messages.append(AIMessage(content=assistant_message))

        print("\n(Patient):")
        for sentence in assistant_message.split('. '):
            if sentence.strip():
                print(sentence.strip() + ('.' if not sentence.endswith('.') else ''))

    return {"chatlog": [m.model_dump() for m in messages]}

# Phase 4: Feedback on interview
def phase_4_feedback(state):
    evaluation_context = load_context('Feedback Criteria')
    instruction_prompt = load_prompt('Feedback')
    chatlog = state["chatlog"]

    # Normalize messages to ensure every dict has a 'role' key
    #This will help format messages later when passing to chatbot
    normalized_chatlog = []
    for m in chatlog:
        if "role" in m:
            role = m["role"]
        elif "type" in m:
            # Map 'type' keys to 'role'
            if m["type"] == "human":
                role = "user"
            elif m["type"] == "ai":
                role = "assistant"
            else:
                role = m["type"]
        else:
            continue
        normalized_chatlog.append({**m, "role": role})

    messages = [
                   SystemMessage(content=f"""{instruction_prompt}\n\n{evaluation_context}"""),
                   HumanMessage(content="Here is the chatlog of the interview. Please provide detailed feedback."),
               ] + [
                   #format chatlog to pass to feedback agent
                   SystemMessage(**m) if m["role"] == "system" else
                   HumanMessage(**m) if m["role"] == "user" else
                   AIMessage(**m)
                   for m in normalized_chatlog
               ]

    response = llm_mini.invoke(messages)
    return {"feedback": response.content}


# Define Graph I/Os
class InterviewState(TypedDict):
    case_study: str                            # Initial input to the workflow
    diagnosis_summary: str                     # Output of phase 1
    clinical_summary: str                      # Output of phase 2
    chatlog: List[Dict[str, Any]]              # Output of phase 3 (list of messages)
    feedback: str                              # Final output from phase 4

# Build LangGraph
workflow = StateGraph(state_schema=InterviewState)
workflow.add_node("profile-verification", RunnableLambda(phase_1_profile_verification))
workflow.add_node("clinical-summary", RunnableLambda(phase_2_summary))
workflow.add_node("roleplay", RunnableLambda(phase_3_roleplay))
workflow.add_node("evaluate", RunnableLambda(phase_4_feedback))

workflow.set_entry_point("profile-verification")
workflow.add_edge("profile-verification", "clinical-summary")
workflow.add_edge("clinical-summary", "roleplay")
workflow.add_edge("roleplay", "evaluate")
workflow.set_finish_point("evaluate")

compiled_graph = workflow.compile()



# Example invocation:

case_study_1 = load_patient_profile('Patient_Profiles/profile_1')
case_study_2 = load_patient_profile('Patient_Profiles/profile_2')

result = compiled_graph.invoke({
     "case_study": case_study_2,
})

print(result["feedback"])