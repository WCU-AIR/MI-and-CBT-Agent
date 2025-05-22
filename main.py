
from LearningBuddy import *


# Initialize LLMs
apiKey = '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'



lb = LearningBuddy(api_key=apiKey)

###Optionally select a case study  ###
#case = load_patient_profile(Patient_Profiles/profile_1)

###Run the workflow ###
result = lb.run(case_study=None)