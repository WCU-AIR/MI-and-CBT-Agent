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



# Modes
STACER_MODE = 1
MI_MODE = 2
CBT_MODE = 3


# Runs a single test with the selected student and patient profile
def run_test(therapy_mode, patient_profile_path, student_profile_number):
    api_key_input = "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"

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


#Extracts the numerical MI scores from the saved session text file
def extract_MI_scores(eval_file_paths):
    # Step 1: Define categories to extract
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
        "Emphasize Autonomy"
    ]

    # Step 2: Initialize dictionary with empty lists
    scores = {cat: [] for cat in categories}

    # Step 3: Loop over files. Use regex to find score categories
    for file_path in eval_file_paths:
        text = Path(file_path).read_text(encoding="utf-8")

        for cat in categories:
            pattern = re.compile(rf"{re.escape(cat)}.*?Score:\s*\**\s*(\d+(\.\d+)?)\**", re.IGNORECASE | re.DOTALL)
            match = pattern.search(text)
            if match:
                score_value = float(match.group(1))
            else:
                score_value = 0.0
            scores[cat].append(score_value)

    # Step 4: Compute overall average per file
    num_files = len(eval_file_paths)
    overall_averages = []
    for i in range(num_files):
        file_scores = [scores[cat][i] for cat in categories]
        overall_avg = round(sum(file_scores) / len(file_scores), 3)
        overall_averages.append(overall_avg)

    # Step 5: Add overall average to dictionary
    scores["Overall Average"] = overall_averages

    return scores

#Plots the bar graph using two dictionaries, one bar for each
def plot_scores(good_student, bad_student):
    # Step 1: Compute averages per category for each set
    categories = list(good_student.keys())
    avg_set1 = [round(np.mean(good_student[cat]), 2) for cat in categories]
    avg_set2 = [round(np.mean(bad_student[cat]), 2) for cat in categories]

    # Replace None with 0 before averaging
    avg_set1 = [round(np.mean([v if v is not None else 0 for v in good_student[cat]]), 2)
                for cat in categories]
    avg_set2 = [round(np.mean([v if v is not None else 0 for v in bad_student[cat]]), 2)
                for cat in categories]

    # Step 2: Plot grouped bar chart
    x = np.arange(len(categories))  # category positions
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, avg_set1, width, label='Good Student', color='skyblue')
    bars2 = ax.bar(x + width / 2, avg_set2, width, label='Bad Student', color='orange')

    # Step 3: Add labels and title
    ax.set_ylabel('Average Score')
    ax.set_xlabel('Score Category')
    ax.set_title('Evaluation Scores for MI')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylim(0, 5.5)
    ax.legend()

    # Step 4: Label bars with values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f'{height}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    return fig



# Runs all the tests in a loop
def run_all_tests(num_patients, student_profiles):
    # Start timer
    start = time.time()

    # Select Patient Profiles
    directory = "Patient_Profiles/MI/"
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if
                 os.path.isfile(os.path.join(directory, f))]
    random_files = random.sample(all_files, num_patients)

    # Select Therapy mode
    mode = MI_MODE

    all_results = []  # store dictionaries for CSV

    scores = []
    df = pd.read_csv('Student_Profiles/AK_StudentAgent_10Profiles - Prompts for MI Students.csv', quotechar='"',
                     skipinitialspace=True)
    # Run all tests
    for student in student_profiles:
        saved_data_directory = []
        for patient in random_files:
            results = run_test(mode, patient, student)
            saved_data_directory.append(results["folder_name"] + "/detailed_feedback.txt")

        student_scores = extract_MI_scores(saved_data_directory)
        scores.append(student_scores)

        # Add student and patient info to each dictionary for CSV
        for idx, patient_file in enumerate(random_files):
            row = {"Student":  df.iloc[student]["Name"], "Patient": os.path.basename(patient_file)}
            # Add all categories from student_scores
            for key, values in student_scores.items():
                row[key] = values[idx] if idx < len(values) else None
            all_results.append(row)

    print("Scores: ", scores)
    fig = plot_scores(scores[0], scores[1])

    # Stop timer
    end = time.time()
    elapsed_seconds = end - start
    elapsed_minutes = elapsed_seconds / 60
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds ({elapsed_minutes:.2f} minutes)")

    # Create a unique folder for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join("saved_tests", f"run_{timestamp}")
    os.makedirs(save_folder, exist_ok=True)

    # Save the figure
    save_plot_path = os.path.join(save_folder, "MI_scores_plot.png")
    fig.savefig(save_plot_path, bbox_inches='tight')
    plt.close(fig)  # close it to free memory
    print(f"Plot saved to: {save_plot_path}")

    # Save elapsed time to a text file
    elapsed_file_path = os.path.join(save_folder, "elapsed_time.txt")
    with open(elapsed_file_path, "w") as f:
        f.write(f"Elapsed time: {elapsed_seconds:.2f} seconds ({elapsed_minutes:.2f} minutes)\n")
    print(f"Elapsed time saved to: {elapsed_file_path}")

    # Save results to CSV
    csv_file_path = os.path.join(save_folder, "MI_scores.csv")
    fieldnames = ["Student", "Patient"] + list(student_scores.keys())
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Results saved to CSV: {csv_file_path}")


#############################################
##Main:

run_all_tests(1, [11,10])
