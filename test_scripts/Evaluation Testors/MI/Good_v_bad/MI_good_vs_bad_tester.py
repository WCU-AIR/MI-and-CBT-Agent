import pandas as pd

import LearningBuddy
import VirtualStudent

import time
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from datetime import datetime
from pathlib import Path
import csv
import json
from dotenv import load_dotenv

load_dotenv()



# Modes
STACER_MODE = 1
MI_MODE = 2
CBT_MODE = 3


# Runs a single test with the selected student and patient profile
def run_test(therapy_mode, patient_profile_path, student_profile_number):
    api_key_input = os.getenv("OPENAI_API_KEY", "")

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

    return session_result


def count_files(directory):
    return sum(
        1 for entry in os.scandir(directory)
        if entry.is_file()
    )

def extract_scores_from_text(text):
    categories = [
        "Acceptance",
        "Empathy",
        "Spirit of MI",
        "Cultivating Change Talk",
        "Softening Sustain Talk",
        "Partnership",
        "Open Questions",
        "Reflections",
        "Affirmations",
        "Persuade with Permission",
        "Emphasize Autonomy",
        "Engaging",
        "Focusing",
        "Evoking",
        "Planning",
        "Summarizing"
    ]

    # Normalize line endings and spacing
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    results = {}

    for cat in categories:
        # Markdown heading pattern, e.g.:
        # ### 1. Acceptance
        # ### Acceptance
        # ### **Acceptance**
        heading_pattern = re.compile(
            rf"^#+\s*(?:\d+\.\s*)?\*?\*?{re.escape(cat)}\*?\*?.*$",
            re.IGNORECASE | re.MULTILINE
        )
        match = heading_pattern.search(text)

        if not match:
            # Heading not found — record as NaN so the Wilcoxon can drop it
            print(f"Could not extract {cat}")
            results[cat] = float('nan')
            continue

        # Extract text starting after this heading until the next ### or end of file
        section_start = match.end()
        next_heading = re.search(r"^###\s", text[section_start:], re.MULTILINE)
        section_end = section_start + next_heading.start() if next_heading else len(text)
        section_text = text[section_start:section_end]

        # Look for Score line — matches patterns like:
        # **Score: 2 (blah)**
        # **Score:** **2**
        # **Score: 2/5**
        score_match = re.search(
            r"\*\*Score[:\s]*\**\s*(\d+(?:\.\d+)?)(?:\s*/\s*\d+)?", section_text, re.IGNORECASE
        )

        if score_match:
            results[cat] = float(score_match.group(1))
        else:
            # Fall back to first standalone number if no explicit Score line
            num_match = re.search(r"(\d+(?:\.\d+)?)", section_text)
            results[cat] = float(num_match.group(1)) if num_match else float('nan')

    # Add Overall Average — skip NaN categories
    numeric_scores = [v for k, v in results.items() if k in categories and v == v]  # v == v is False for NaN
    results["Overall Average"] = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else float('nan')

    return results

def extract_all_scores_from_folder(parent_folder, filename="detailed_feedback.txt"):
    all_results = []
    for subdir in os.listdir(parent_folder):
        subpath = os.path.join(parent_folder, subdir)
        if os.path.isdir(subpath):
            file_path = os.path.join(subpath, filename)
            if os.path.exists(file_path):
                #print(f"Extracting scores from {subpath}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                scores = extract_scores_from_text(text)
                scores["session_folder"] = subdir
                all_results.append(scores)
            else:
                print(f"⚠️ File not found in: {subpath}")
    return pd.DataFrame(all_results)

# Runs all the tests in a loop
def run_all_tests(num_patients, student_profiles):
    start = time.time()

    # Select Patient Profiles
    directory = "Patient_Profiles/MI/"
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    random_files = random.sample(all_files, num_patients)

    mode = MI_MODE
    all_results = []
    path = 'Student_Profiles/StudentAgent_Profiles - Prompts for MI Students.csv'
    df = pd.read_csv(path, quotechar='"', skipinitialspace=True)

    # Create save folder
    base_folder = "saved_tests"
    os.makedirs(base_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(base_folder, f"run_{timestamp}")
    os.makedirs(save_folder, exist_ok=True)

    # Load checkpoint if it exists
    checkpoint_path = os.path.join(save_folder, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        completed_pairs = set(tuple(x) for x in checkpoint["completed_pairs"])
        all_file_paths = checkpoint["all_file_paths"]
    else:
        completed_pairs = set()
        all_file_paths = [[] for _ in student_profiles]

    num_tests = len(student_profiles) * len(random_files)
    count = 1

    student_number = 0
    # Run all tests
    for student in student_profiles:
        for patient in random_files:
            pair_key = (student, patient)
            if pair_key in completed_pairs:
                print(f"Skipping completed pair: Student {student}, Patient {os.path.basename(patient)}")
                count += 1
                continue

            print(f"Running test #{count} of {num_tests}...")
            try:
                results = run_test(mode, patient, student)
                feedback_path = results["folder_name"] + "/detailed_feedback.txt"
                all_file_paths[student_number].append(feedback_path)

                completed_pairs.add(pair_key)

                # Save checkpoint
                checkpoint_data = {
                    "completed_pairs": list(completed_pairs),
                    "all_file_paths": all_file_paths
                }
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=2)

            except Exception as e:
                print(f"Error during pair (Student {student}, Patient {patient}): {e}")
                continue

            count += 1
        student_number += 1

    # Once all tests complete, run regex step only once
    parent_folder = "Saved_Sessions"
    df = extract_all_scores_from_folder(parent_folder)

    # Add "Student" column
    half = len(df) // 2
    df['Student'] = ['good student'] * half + ['bad student'] * (len(df) - half)

    # Save results to CSV
    output_csv = os.path.join(save_folder, "extracted_scores.csv")
    df.to_csv(output_csv, index=False)

    # Save elapsed time
    end = time.time()
    elapsed_minutes = (end - start) / 60
    with open(os.path.join(save_folder, "elapsed_time.txt"), "w") as f:
        f.write(f"Elapsed time: {elapsed_minutes:.2f} minutes\n")

    print(f"Run complete. Checkpoint and results saved in: {save_folder}")

#############################################
##Main:
num_patients = count_files("Patient_Profiles/MI/")
print(num_patients)

run_all_tests(num_patients, [1, 0])





