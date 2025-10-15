# import LearningBuddy
# import VirtualStudent

# import time

# # Modes
# STACER_MODE = 1
# MI_MODE = 2
# CBT_MODE = 3


# def run_test(therapy_mode, patient_profile_path, student_profile_number):
#     api_key_input = "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"

#     #Start timer
#     start = time.time()

#     # Create lb object
#     if therapy_mode == MI_MODE:
#         lb=LearningBuddy.LearningBuddy(api_key_input, "null", LearningBuddy.LearningBuddy.MI_MODE)
#     elif therapy_mode == CBT_MODE:
#         lb=LearningBuddy.LearningBuddy(api_key_input, "null", LearningBuddy.LearningBuddy.CBT_MODE)

#     #Load student profile for student agent
#     if therapy_mode == MI_MODE:
#         student = VirtualStudent.VirtualStudent(api_key_input, VirtualStudent.VirtualStudent.MI, student_num=student_profile_number)
#     elif therapy_mode == CBT_MODE:
#         student = VirtualStudent.VirtualStudent(api_key_input, VirtualStudent.VirtualStudent.CBT, student_num=student_profile_number)

#     # Load patient profile for LB patient agent
#     case = lb.load_patient_profile(patient_profile_path)

#     # Initialize LB
#     session_result = lb.start_session(case_study=case)

#     #Loop for student agent and patient agent to talk to each other-- limit of 10
#     for i in range(0, 10):
#         student_utterance = student.get_response(session_result["chatlog"])
#         session_result = lb.send_message(student_utterance)

#     #generate Feedback:
#     session_result = lb.end_session()

#     #Everything gets saved to Saved Session Folder

#     # Stop timer
#     end = time.time()
#     elapsed = end - start
#     print(elapsed)


# #############################################
# ##Main:

# #Select Student Profile
# case_filepath = "Patient_Profiles/STACER/Antisocial Personality Disorder.txt"

# #Select Student Profile
# student_num = 11 # student num is the row in the csv file for the patient you want to use

# # Select Therapy mode
# mode = CBT_MODE

# #Run Tests -- Results are saved to Saved_Sessions
# run_test(mode, case_filepath, student_num)

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# =====================
# READ AND PREPARE DATA
# =====================
df = pd.read_csv("saved_tests/run_20251015_104140/CBT_scores.csv")

# Select students
good_student_name = "12_Expert_CBTVirtuoso"
bad_student_name = "11_Novice_UndirectedAdviceGiver"

good_df = df[df["Student"] == good_student_name]
bad_df = df[df["Student"] == bad_student_name]

# Extract numeric score columns (exclude metadata)
score_cols = [c for c in df.columns if c not in ["Student", "Patient", "Overall Average"]]

# Build category dictionaries
good_student = {col: good_df[col].dropna().tolist() for col in score_cols}
bad_student  = {col: bad_df[col].dropna().tolist()  for col in score_cols}

# =====================
# WILCOXON SIGNED-RANK TEST
# =====================
print("Wilcoxon signed-rank test (paired, non-parametric):\n")

results = []
for cat in score_cols:
    expert_vals = np.array(good_student[cat])
    novice_vals = np.array(bad_student[cat])

    # Ensure both have same number of observations
    n = min(len(expert_vals), len(novice_vals))
    if n > 0:
        stat, p = wilcoxon(expert_vals[:n], novice_vals[:n])
        results.append((cat, stat, p))

# Sort by p-value
results.sort(key=lambda x: x[2])

# Print results
for cat, stat, p in results:
    significance = "⭐" if p < 0.05 else ""
    print(f"{cat:45s}  stat={stat:7.7f}  p={p:.7f} {significance}")
