import LearningBuddy
import VirtualStudent

import time

# Modes
STACER_MODE = 1
MI_MODE = 2
CBT_MODE = 3


def run_test(therapy_mode, patient_profile_path, student_profile_number):
    api_key_input = " ADD API KEY HERE"

    #Start timer
    start = time.time()

    # Create lb object
    if therapy_mode == MI_MODE:
        lb=LearningBuddy.LearningBuddy(api_key_input, "null", LearningBuddy.LearningBuddy.MI_MODE)
    elif therapy_mode == CBT_MODE:
        lb=LearningBuddy.LearningBuddy(api_key_input, "null", LearningBuddy.LearningBuddy.CBT_MODE)

    #Load student profile for student agent
    if therapy_mode == MI_MODE:
        student = VirtualStudent.VirtualStudent(api_key_input, VirtualStudent.VirtualStudent.MI, student_num=student_profile_number)
    elif therapy_mode == CBT_MODE:
        student = VirtualStudent.VirtualStudent(api_key_input, VirtualStudent.VirtualStudent.CBT, student_num=student_profile_number)

    # Load patient profile for LB patient agent
    case = lb.load_patient_profile(patient_profile_path)

    # Initialize LB
    session_result = lb.start_session(case_study=case)

    #Loop for student agent and patient agent to talk to each other-- limit of 10
    for i in range(0, 10):
        student_utterance = student.get_response(session_result["chatlog"])
        session_result = lb.send_message(student_utterance)

    #generate Feedback:
    session_result = lb.end_session()

    #Everything gets saved to Saved Session Folder

    # Stop timer
    end = time.time()
    elapsed = end - start
    print(elapsed)


#############################################
##Main:

#Select Student Profile
case_filepath = "Patient_Profiles/MI/AdrianCole.txt"

#Select Student Profile
student_num = 0 # student num is the row in the csv file for the patient you want to use

# Select Therapy mode
mode = MI_MODE

#Run Tests -- Results are saved to Saved_Sessions
run_test(mode, case_filepath, student_num)