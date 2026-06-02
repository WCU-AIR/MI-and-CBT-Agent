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

def run_test(therapy_mode, patient_profile_path, student_profile_number, prompt_addition = ""):
    api_key_input = os.getenv("OPENAI_API_KEY", "")

    # Create lb object
    if therapy_mode == MI_MODE:
        lb=LearningBuddy.LearningBuddy(api_key_input, "null", LearningBuddy.LearningBuddy.MI_MODE)
    elif therapy_mode == CBT_MODE:
        lb=LearningBuddy.LearningBuddy(api_key_input, "null", LearningBuddy.LearningBuddy.CBT_MODE)

    #Load student profile for student agent
    if therapy_mode == MI_MODE:
        student = VirtualStudent.VirtualStudent(api_key_input, VirtualStudent.VirtualStudent.MI, student_num=student_profile_number, prompt_add_on=prompt_addition)
    elif therapy_mode == CBT_MODE:
        student = VirtualStudent.VirtualStudent(api_key_input, VirtualStudent.VirtualStudent.CBT, student_num=student_profile_number, prompt_add_on=prompt_addition)

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

def read_file(file_path: str) -> str:

    return Path(file_path).read_text(encoding="utf-8", errors="ignore")

def plot_scores_from_csv(csv_file):
    # Step 1: Load CSV
    df = pd.read_csv(csv_file)

    # Step 2: Identify score columns
    score_columns = df.select_dtypes(include=['number']).columns

    # Step 3: Group by 'Student' and calculate mean (replace None/NaN with 0)
    df[score_columns] = df[score_columns].fillna(0)
    grouped = df.groupby('student')[score_columns].mean()

    categories = score_columns
    good_scores = grouped.loc['first try'].tolist()
    bad_scores = grouped.loc['second try'].tolist()

    # Step 4: Plot grouped bar chart
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, good_scores, width, label='First Attempt', color='skyblue')
    bars2 = ax.bar(x + width / 2, bad_scores, width, label='Second Attempt', color='orange')

    ax.set_ylabel('Average Score')
    ax.set_xlabel('Score Category')
    ax.set_title('Evaluation Scores for MI')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylim(0, 5.5)
    ax.legend()

    # Step 5: Label bars with values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig

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
            # Some headings might include parentheses or extra symbols
            print(f"Could not extract {cat}")
            results[cat] = 1.0
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
            results[cat] = float(num_match.group(1)) if num_match else 1.0

    # Add Overall Average
    numeric_scores = [v for k, v in results.items() if k in categories]
    results["Overall Average"] = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else 1.0

    return results

def extract_recommendations_section(text_or_path: str) -> str:
    """
    Extracts the 'Recommendations' section from text or a file.
    Stops when the next header (## ...) or the end of the document is reached.
    """

    # Detect whether argument is a path or full text
    if Path(text_or_path).exists():
        text = Path(text_or_path).read_text(encoding="utf-8", errors="ignore")
    else:
        text = text_or_path

    # Match the Recommendations section until the next header (## ...)
    pattern = re.compile(
        r"(?:#+\s*\**\s*Recommendations\s*\**:?\s*)(.*?)(?=\n\s*#+\s|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    match = pattern.search(text)
    if match:
        section = match.group(1).strip()

        # Remove trailing markdown dividers (like --- or ***)
        section = re.sub(r"^[-*_]{3,}\s*$", "", section, flags=re.MULTILINE).strip()

        return section

    return ""


def run_tests(num_tests):
    checkpoint_path = "saved_tests/first_vs_second/checkpoint.json"
    os.makedirs("saved_tests/first_vs_second", exist_ok=True)

    # Select Bad Student Second Try
    student_profile = 14

    # Walk through the directory tree and collect subdirectories from the first run
    base_directory = "Saved_Sessions_good_v_bad/Bad"
    subdirs = []
    for root, dirs, files in os.walk(base_directory):
        for d in sorted(dirs):
            full_path = os.path.join(root, d)
            subdirs.append(full_path)

    # -----------------------------
    # Load checkpoint (if exists)
    # -----------------------------
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        completed_sessions = set(checkpoint.get("completed_sessions", []))
    else:
        completed_sessions = set()
        checkpoint = {"completed_sessions": [], "csv_path": ""}

    # If CSV already exists, load partial progress
    csv_path = os.path.join("saved_tests/first_vs_second", "student_scores.csv")
    if os.path.exists(csv_path):
        df_all = pd.read_csv(csv_path)
    else:
        df_all = pd.DataFrame()

    # -----------------------------
    # Main loop
    # -----------------------------
    for i in range(num_tests):
        print(f"Test {i}")
        first_session = subdirs[i]
        session_name = os.path.basename(first_session)

        # Skip completed sessions
        if session_name in completed_sessions:
            print(f"Skipping {session_name} (already completed).")
            continue

        print(f"Running session {session_name}...")

        patient_profile = os.path.join(first_session, "case_study.txt")

        # Extract scores from first run
        results_text = read_file(os.path.join(first_session, "detailed_feedback.txt"))
        scores = extract_scores_from_text(results_text)

        # Extract recommendations
        recommendations = (extract_recommendations_section(results_text))

        df_first = pd.DataFrame([scores])
        df_first["student"] = "first try"
        df_first["session"] = session_name

        # Second Run
        results_run_2 = run_test(MI_MODE, patient_profile, student_profile, recommendations)
        results_text_run_2 = results_run_2["feedback"]
        scores_run_2 = extract_scores_from_text(results_text_run_2)

        df_second = pd.DataFrame([scores_run_2])
        df_second["student"] = "second try"
        df_second["session"] = session_name

        # Combine into main dataframe and save incrementally
        df_all = pd.concat([df_all, df_first, df_second], ignore_index=True)
        df_all.to_csv(csv_path, index=False)

        # Update checkpoint
        completed_sessions.add(session_name)
        checkpoint["completed_sessions"] = list(completed_sessions)
        checkpoint["csv_path"] = csv_path
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        print(f"✅ Completed session {session_name}. Progress saved.")

    # -----------------------------
    # After all runs, plot results
    # -----------------------------
    figure = plot_scores_from_csv(csv_path)
    figure_path = os.path.join("saved_tests/first_vs_second", "scores_plot.png")
    figure.savefig(figure_path, bbox_inches="tight")

    print(f"\nAll done! Results saved to {csv_path}")

run_tests(133)