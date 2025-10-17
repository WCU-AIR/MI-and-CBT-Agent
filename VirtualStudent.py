import openai
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import pandas as pd



class VirtualStudent:
    #Therapy Types
    STACER = 1
    MI = 2
    CBT = 3
    def __init__(self, api_key: str, therapy_type, student_num=0, prompt_add_on = ""):
        self.api_key = SecretStr(api_key)
        self.therapy_type = therapy_type
        self.student_num = student_num
        self.llm_mini_virtualStudent = ChatOpenAI(model="gpt-4.1", temperature=1, api_key=self.api_key)

        self.student_prompt = self._load_full_prompt() + prompt_add_on
        print(self.student_prompt)

        #self.presentation_prompt = 'Student Presentation'
        #self.present_filepath = self._load_filepath(self.presentation_prompt)
        #self.presentation_prompt = self._load_prompt(self.present_filepath)


    def _load_full_prompt(self) -> str:
        if self.therapy_type == self.MI:
            df = pd.read_csv('Student_Profiles/AK_StudentAgent_10Profiles - Prompts for MI Students.csv', quotechar='"', skipinitialspace=True)
            print("Student Profile Selected: ", df.iloc[self.student_num]["Name"])
            prompt = self._load_prompt("Prompts/MI/virtual_student_prompt.txt")
            profile  = (df.iloc[self.student_num]["Prompt"])
            final_prompt = prompt + profile
            #print("Full Prompt: ", final_prompt)
            return final_prompt
        elif self.therapy_type == self.CBT:
            df = pd.read_csv('Student_Profiles/AK_StudentAgent_10Profiles - Prompts for CBT Students.csv', quotechar='"', skipinitialspace=True)
            print("Student Profile Selected: ", df.iloc[self.student_num]["Name"])
            prompt = self._load_prompt("Prompts/CBT/virtual_student_prompt.txt")
            profile = (df.iloc[self.student_num]["Prompt"])
            final_prompt = prompt + profile
            #print("Full Prompt: ", final_prompt)
            return final_prompt
        else:
            return " "

    def _load_prompt(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def _extract_clean_chatlog(self, chatlog) :
        cleaned = []
        for entry in chatlog:
            role = entry.get("role")
            content = entry.get("content", "").strip()

            # Skip system messages
            if role == "system":
                continue

            # Skip the big case file or prompt
            if role == "user" and "# **Prompt:" in content:
                continue

            cleaned.append(entry)
        return cleaned

    def get_response(self, conversation):
        conversation = self._extract_clean_chatlog(conversation)
        message = [f"{self.student_prompt}\n\n{conversation}"]
        response = self.llm_mini_virtualStudent.invoke(message)
        return response.content

  # def present_casesummary(self, conversation):
  #     message = [f"{self.presentation_prompt}\n\n{conversation}"]
  #     response = self.llm_mini_virtualStudent.invoke(message)
  #     return response.content