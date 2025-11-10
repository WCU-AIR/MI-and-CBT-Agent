import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import VirtualStudent
from LearningBuddy_Agent import LearningBuddy, MI_MODE, CBT_MODE
from utils.api_key_loader import load_openai_api_key

# =============================
# === Experiment Constants ====
# =============================

STUDENT_CSV_PATH = "Student_Profiles/AK_StudentAgent_10Profiles - Prompts for CBT Students.csv"
PATIENT_DIR_MI = "Patient_Profiles/MI/"
PATIENT_DIR_CBT = "Patient_Profiles/CBT/"

GOOD_STUDENT_NUM = 11
BAD_STUDENT_NUM = 10

API_KEY = load_openai_api_key()


# =============================
# === Core Helper Functions ===
# =============================

def get_patient_files(mode, start_idx=None, end_idx=None):
    """Get patient files in alphabetical order, optionally sliced by range."""
    patient_dir = PATIENT_DIR_MI if mode == 'MI' else PATIENT_DIR_CBT

    all_files = sorted([
        os.path.join(patient_dir, f)
        for f in os.listdir(patient_dir)
        if os.path.isfile(os.path.join(patient_dir, f))
    ])

    if start_idx is not None and end_idx is not None:
        return all_files[start_idx:end_idx]
    return all_files


def run_test(therapy_mode, patient_profile_path, student_profile_number,
             patient_number=None, student_type=None, student_feedback_csv="", api_key_input=None):
    """Run one simulated therapy session between student and patient.

    Args:
        therapy_mode: 'MI' or 'CBT'
        patient_profile_path: Path to patient profile file
        student_profile_number: Index of student profile to use
        patient_number: Optional patient identifier for folder naming
        student_type: Optional student type ('good', 'bad_1st', 'bad_feedback') for folder naming
        api_key_input: Optional API key
    """
    api_key_input = API_KEY or "key"

    modes = {
        'MI': MI_MODE,
        'CBT': CBT_MODE
    }

    # Prepare session metadata for folder naming
    session_metadata = {}
    if patient_number is not None:
        session_metadata['patient_number'] = patient_number
    if student_type is not None:
        session_metadata['student_type'] = student_type

    add_on = ""
    if not student_feedback_csv == "":
        df = pd.read_csv(student_feedback_csv, quotechar='"', skipinitialspace=True)
        add_on = df.loc[student_profile_number, "Prompt"]

    # Initialize LearningBuddy with appropriate mode
    if modes[therapy_mode] == MI_MODE:
        lb = LearningBuddy(api_key_input, "null", MI_MODE)
        student = VirtualStudent.VirtualStudent(
            api_key_input,
            VirtualStudent.VirtualStudent.MI,
            student_index=student_profile_number,
            prompt_addon=add_on
        )
    elif modes[therapy_mode] == CBT_MODE:
        lb = LearningBuddy(api_key_input, "null", CBT_MODE)
        student = VirtualStudent.VirtualStudent(
            api_key_input,
            VirtualStudent.VirtualStudent.CBT,
            student_index=student_profile_number,
            prompt_addon=add_on
        )
    else:
        raise ValueError("Invalid therapy mode selected")

    # Load patient profile and start session with metadata
    case = lb.load_patient_profile(patient_profile_path)

    # Pass metadata to start_session
    session_result = lb.start_session(case_study=case, session_metadata=session_metadata)

    # Run 10 exchanges between student and patient
    for _ in range(10):
        current_state = lb.get_session_state()
        interview_messages = current_state.get("interview_messages", [])
        student_utterance = student.get_response(interview_messages)
        session_result = lb.send_message(student_utterance)

    # End session and get feedback
    session_result = lb.end_session()
    final_state = lb.get_session_state()

    # Get folder name - try both possible keys
    folder_name = final_state.get("export_folder") or final_state.get("folder_name", "")

    return {
        "session_result": session_result,
        "folder_name": folder_name
    }

def extract_scores(eval_file_paths, mode):
    """Extract numerical scores from feedback text files based on therapy mode."""

    # Define categories based on therapy mode
    if mode == "MI":
        categories = [
            "Cultivating Change Talk",
            "Softening Sustain Talk",
            "Partnership",
            "Empathy",
            "Acceptance",
            "Direction",
            "Autonomy Support",
            "Collaboration",
            "Evocation"
        ]
    elif mode == "CBT":
        categories = [
            "Agenda",
            "Feedback",
            "Understanding",
            "Interpersonal Effectiveness",
            "Collaboration",
            "Pacing and Efficient Use of Time",
            "Guided Discovery",
            "Focusing on Key Cognitions or Behaviors",
            "Strategy for Change",
            "Application of CBT Techniques",
            "Homework"
        ]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'MI' or 'CBT'")

    scores = {cat: [] for cat in categories}

    for file_path in eval_file_paths:
        text = Path(file_path).read_text(encoding="utf-8")

        for cat in categories:
            pattern = re.compile(
                rf"{re.escape(cat)}.*?Score:\s*\**\s*(\d+(\.\d+)?)\**",
                re.IGNORECASE | re.DOTALL
            )
            matches = pattern.findall(text)
            if matches:
                score_value = float(matches[-1][0])
            else:
                score_value = 0.0
            scores[cat].append(score_value)

    num_files = len(eval_file_paths)
    overall_averages = []
    for i in range(num_files):
        file_scores = [scores[cat][i] for cat in categories]
        overall_avg = round(sum(file_scores) / len(file_scores), 3)
        overall_averages.append(overall_avg)

    scores["Overall Average"] = overall_averages
    return scores


def extract_recommendations(feedback_files):
    """
    Extracts the 'Recommendations' section content, stopping cleanly
    before the subsequent 'Overall Session Score' or separator.
    """
    recommendations = []

    # Define the pattern to find the Recommendations section.
    # 1. Looks for the word "Recommendations" (case-insensitive).
    # 2. Uses a non-greedy match (.*?) to capture the content following it.
    # 3. Uses a lookahead assertion (?=...) to define the end boundary
    #    without including it in the captured group. The boundary is
    #    a horizontal rule (---) or the heading "## Overall Session Score".
    pattern = re.compile(
        r"Recommendations\s*(.*?)(?=\n\s*---|\n\s*## Overall Session Score)",
        re.DOTALL | re.IGNORECASE
    )

    for path in feedback_files:
        try:
            # Assumes 'path' points to a file that can be read.
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Error: File not found at {path}")
            continue
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            continue

        matches = pattern.findall(text)

        if matches:
            # The desired content is in the first (and only) capturing group,
            # so we take the last match's content and clean up whitespace.
            recommendations.append(matches[-1].strip())

    return recommendations


def augment_student_prompt(student_csv, student_num, feedback_texts, save_as="Student_Profiles/temp_student.csv"):
    """Append combined feedback to student's prompt and save as new CSV."""
    df = pd.read_csv(student_csv, quotechar='"', skipinitialspace=True)
    combined_feedback = "\n\n---\n\n".join(feedback_texts)
    improvement_intro = (
        "Now, reflect on the weaknesses of the previous instructions and strive to enhance your skills and performance during the therapy session with the patient by strictly incorporating the list of recommendations. "
        "End the conversation after exactly 10 clinician–patient exchanges. \n\n"
    )

    df.loc[student_num, "Prompt"] += f"\n\n# Feedback to incorporate:\n{improvement_intro}" \
                                     f"{combined_feedback}"
    df.to_csv(save_as, index=False)
    return save_as


# =============================
# === Experiment Pipeline ====
# =============================

def run_experiment(student_num, patient_files, student_csv_path, mode, label, save_folder):
    """Run experiment for one student configuration."""
    df = pd.read_csv(student_csv_path, quotechar='"', skipinitialspace=True)
    all_results = []
    saved_data_directory = []

    # Run tests for each patient
    for patient_idx, patient in enumerate(patient_files):
        # Extract patient number from filename or use index
        patient_basename = os.path.basename(patient)
        patient_number = patient_idx + 1  # Or extract from filename if it contains a number

        results = run_test(
            mode,
            patient,
            student_num,
            patient_number=patient_number,
            student_type=label
        )
        feedback_path = results["folder_name"] + f"/detailed_feedback_{label}.txt"
        saved_data_directory.append(feedback_path)

    # Extract scores from all feedback files
    student_scores = extract_scores(saved_data_directory, mode)

    # Build results table
    for idx, patient_file in enumerate(patient_files):
        row = {
            "Student": df.iloc[student_num]["Name"],
            "Patient": os.path.basename(patient_file)
        }
        for key, values in student_scores.items():
            row[key] = values[idx] if idx < len(values) else None
        all_results.append(row)

    # Save results
    csv_path = os.path.join(save_folder, f"{label}_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Student", "Patient"] + list(student_scores.keys())
        )
        writer.writeheader()
        writer.writerows(all_results)

    return student_scores, saved_data_directory


def run_parallel_experiments(args):
    STUDENT_CSV_PATH = "Student_Profiles/AK_StudentAgent_10Profiles - Prompts for CBT Students.csv" if args.mode == "CBT" else "Student_Profiles/AK_StudentAgent_10Profiles - Prompts for MI Students.csv"
    """Main function for parallel execution mode."""
    start_time = time.time()

    # Get patient files for this range
    patient_files = get_patient_files(args.mode, args.start_patient, args.end_patient)
    num_patients = len(patient_files)

    print(f"Processing patients {args.start_patient} to {args.end_patient} ({num_patients} patients)")
    print(f"Mode: {args.mode}")

    # Create save folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(
        "saved_tests",
        f"{args.run_id}_patients_{args.start_patient}_{args.end_patient}_{timestamp}"
    )
    os.makedirs(save_folder, exist_ok=True)

    # Save metadata
    metadata = {
        "run_id": args.run_id,
        "mode": args.mode,
        "start_patient": args.start_patient,
        "end_patient": args.end_patient,
        "num_patients": num_patients,
        "timestamp": timestamp
    }
    with open(os.path.join(save_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Initialize storage for all results
    all_feedback_files_bad1 = []
    all_results_good = []
    all_results_bad1 = []
    all_results_feedback = []

    df = pd.read_csv(STUDENT_CSV_PATH, quotechar='"', skipinitialspace=True)

    # Process each patient with all three student types
    for patient_idx, patient_file in enumerate(patient_files):
        print(f"\n{'=' * 60}")
        print(f"Processing Patient {patient_idx + 1}/{num_patients}: {os.path.basename(patient_file)}")
        print(f"{'=' * 60}")

        patient_number = patient_idx + 1

        # 1️⃣ Good student for this patient
        print(f"  [STEP 1] Running Good Student...")
        results_good = run_test(
            args.mode,
            patient_file,
            GOOD_STUDENT_NUM,
            patient_number=patient_number,
            student_type="good_student"
        )
        # Look for the feedback file - it might be named differently
        folder_name = results_good["folder_name"]

        # Try to find the feedback file (check different possible names)
        possible_names = [
            "detailed_feedback_good_student.txt",
            "detailed_feedback.txt",
            "feedback.txt"
        ]

        feedback_path_good = None
        for name in possible_names:
            potential_path = os.path.join(folder_name, name)
            if os.path.exists(potential_path):
                feedback_path_good = potential_path
                break

        if feedback_path_good is None:
            # List all files in the folder to see what's actually there
            print(f"  Warning: Could not find feedback file in {folder_name}")
            print(
                f"  Files in folder: {os.listdir(folder_name) if os.path.exists(folder_name) else 'folder does not exist'}")
            # Use a default pattern - adjust based on what you see
            feedback_path_good = os.path.join(folder_name, "detailed_feedback.txt")

        scores_good = extract_scores([feedback_path_good], args.mode)

        # Store good student result
        row_good = {
            "Student": df.iloc[GOOD_STUDENT_NUM]["Name"],
            "Patient": os.path.basename(patient_file)
        }
        for key, values in scores_good.items():
            row_good[key] = values[0] if values else None
        all_results_good.append(row_good)

        # 2️⃣ Bad student (1st attempt) for this patient
        print(f"  [STEP 2] Running Bad Student (1st attempt)...")
        results_bad1 = run_test(
            args.mode,
            patient_file,
            BAD_STUDENT_NUM,
            patient_number=patient_number,
            student_type="bad_student_1st"
        )
        folder_name_bad1 = results_bad1["folder_name"]

        # Find feedback file for bad student
        feedback_path_bad1 = None
        for name in possible_names:
            potential_path = os.path.join(folder_name_bad1, name)
            if os.path.exists(potential_path):
                feedback_path_bad1 = potential_path
                break

        if feedback_path_bad1 is None:
            feedback_path_bad1 = os.path.join(folder_name_bad1, "detailed_feedback.txt")

        all_feedback_files_bad1.append(feedback_path_bad1)
        scores_bad1 = extract_scores([feedback_path_bad1], args.mode)

        # Store bad student 1st result
        row_bad1 = {
            "Student": df.iloc[BAD_STUDENT_NUM]["Name"],
            "Patient": os.path.basename(patient_file)
        }
        for key, values in scores_bad1.items():
            row_bad1[key] = values[0] if values else None
        all_results_bad1.append(row_bad1)

        # 3️⃣ Bad student with feedback for this patient
        print(f"  [STEP 3] Running Bad Student with Feedback...")
        feedback_text = extract_recommendations([feedback_path_bad1])
        updated_csv = augment_student_prompt(
            STUDENT_CSV_PATH,
            BAD_STUDENT_NUM,
            feedback_text,
            save_as=os.path.join(save_folder, f"temp_student_feedback_{patient_idx}.csv")
        )

        results_feedback = run_test(
            args.mode,
            patient_file,
            BAD_STUDENT_NUM,
            patient_number=patient_number,
            student_type="bad_student_feedback",
            student_feedback_csv=updated_csv
        )
        folder_name_feedback = results_feedback["folder_name"]

        # Find feedback file
        feedback_path_feedback = None
        for name in possible_names:
            potential_path = os.path.join(folder_name_feedback, name)
            if os.path.exists(potential_path):
                feedback_path_feedback = potential_path
                break

        if feedback_path_feedback is None:
            feedback_path_feedback = os.path.join(folder_name_feedback, "detailed_feedback.txt")

        scores_feedback = extract_scores([feedback_path_feedback], args.mode)

        # Store bad student with feedback result
        row_feedback = {
            "Student": df.iloc[BAD_STUDENT_NUM]["Name"] + " (with feedback)",
            "Patient": os.path.basename(patient_file)
        }
        for key, values in scores_feedback.items():
            row_feedback[key] = values[0] if values else None
        all_results_feedback.append(row_feedback)

    # Save all results to CSV files
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    # Determine score columns from first result
    score_cols = [k for k in all_results_good[0].keys() if k not in ["Student", "Patient"]]

    # Save good student scores
    csv_path_good = os.path.join(save_folder, "good_student_scores.csv")
    with open(csv_path_good, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Student", "Patient"] + score_cols)
        writer.writeheader()
        writer.writerows(all_results_good)

    # Save bad student 1st scores
    csv_path_bad1 = os.path.join(save_folder, "bad_student_1st_scores.csv")
    with open(csv_path_bad1, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Student", "Patient"] + score_cols)
        writer.writeheader()
        writer.writerows(all_results_bad1)

    # Save bad student feedback scores
    csv_path_feedback = os.path.join(save_folder, "bad_student_feedback_scores.csv")
    with open(csv_path_feedback, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Student", "Patient"] + score_cols)
        writer.writeheader()
        writer.writerows(all_results_feedback)

    # Save elapsed time
    elapsed_seconds = time.time() - start_time
    with open(os.path.join(save_folder, "elapsed_time.txt"), "w") as f:
        f.write(f"Elapsed time: {elapsed_seconds:.2f} seconds\n")

    print(f"\n✅ Parallel run complete! Results saved to {save_folder}")
    return save_folder


def plot_scores(score_sets, labels, title, colors=None, ylim=(0, 6.0)):
    """Plot bar chart comparing multiple sets of scores with error bars."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 14

    if colors is None:
        colors = ['skyblue', 'orange', 'green', 'red']

    # Reorder the data: Novice → Novice+Feedback → Expert
    # Find indices for reordering
    reorder_map = []
    for i, label in enumerate(labels):
        if "Novice Student + Feedback" in label or "After" in label:
            reorder_map.append((i, 1))  # Second position
        elif "Expert" in label or "Good" in label:
            reorder_map.append((i, 2))  # Third position
        elif "Novice" in label or "Bad" in label or "Before" in label:
            reorder_map.append((i, 0))  # First position
        else:
            reorder_map.append((i, i))  # Keep original position if no match

    # Sort by target position
    reorder_map.sort(key=lambda x: x[1])
    indices = [x[0] for x in reorder_map]

    # Reorder all lists
    score_sets = [score_sets[i] for i in indices]
    labels = [labels[i] for i in indices]
    colors = [colors[i] for i in indices]

    categories = list(score_sets[0].keys())
    num_sets = len(score_sets)

    # Calculate averages and standard deviations
    averages = []
    std_devs = []
    for scores in score_sets:
        avg = []
        std = []
        for cat in categories:
            values = [v if v is not None else 0 for v in scores[cat]]
            avg.append(round(np.mean(values), 2))
            std.append(round(np.std(values, ddof=1) if len(values) > 1 else 0, 2))
        averages.append(avg)
        std_devs.append(std)

    x = np.arange(len(categories))

    # --- MODIFICATION 2: Dynamic bar width to shrink whitespace between categories ---
    # Target to use about 85% of the available space (1.0 unit) to leave a small gap (15% gap)
    space_to_use = 0.85
    width = space_to_use / num_sets
    bar_width = width

    # Calculate figure width based on number of categories to maintain consistent spacing
    fig_width = max(14, len(categories) * num_sets * 0.8 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars_list = []
    for i, (avg, std, label, color) in enumerate(zip(averages, std_devs, labels, colors[:num_sets])):
        offset = (i - num_sets / 2 + 0.5) * width
        bars = ax.bar(x + offset, avg, width, label=label, color=color,
                      yerr=std, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})
        bars_list.append(bars)

    ax.set_ylabel('Average Score')
    ax.set_xlabel('Score Category')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')

    # --- MODIFICATION 1: Use the new ylim parameter for consistent Y-axis ---
    ax.set_ylim(ylim)
    ax.legend(loc='upper left')

    # Add value labels in the middle of bars with 11pt Arial font
    for bars, avg_vals in zip(bars_list, averages):
        for bar, height in zip(bars, avg_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,  # Position in the middle of the bar
                f'{height:.1f}',
                ha='center',
                va='center',  # Center vertically
                fontsize=14,  # Changed to 11pt
                fontfamily='Arial'  # Explicit Arial font
            )

    plt.tight_layout()
    return fig
def aggregate_results(args):
    """Aggregate results from multiple parallel runs and generate plots."""
    print(f"\n{'=' * 60}")
    print(f"Aggregating results for run_id: {args.run_id}")
    print(f"{'=' * 60}")

    # Find all folders for this run_id
    pattern = os.path.join("saved_tests", f"{args.run_id}_patients_*")
    folders = glob(pattern)

    if not folders:
        print(f"❌ No folders found for run_id: {args.run_id}")
        return

    print(f"Found {len(folders)} parallel run folders")

    # Load and combine all CSVs
    good_dfs = []
    bad1_dfs = []
    bad_feedback_dfs = []

    for folder in sorted(folders):
        print(f"  Loading from {os.path.basename(folder)}...")
        good_dfs.append(pd.read_csv(os.path.join(folder, "good_student_scores.csv")))
        bad1_dfs.append(pd.read_csv(os.path.join(folder, "bad_student_1st_scores.csv")))
        bad_feedback_dfs.append(pd.read_csv(os.path.join(folder, "bad_student_feedback_scores.csv")))

    # Combine dataframes
    df_good = pd.concat(good_dfs, ignore_index=True)
    df_bad1 = pd.concat(bad1_dfs, ignore_index=True)
    df_bad_feedback = pd.concat(bad_feedback_dfs, ignore_index=True)

    # Create aggregated folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_folder = os.path.join("saved_tests", f"{args.run_id}_aggregated_{timestamp}")
    os.makedirs(agg_folder, exist_ok=True)

    # Save combined CSVs
    df_good.to_csv(os.path.join(agg_folder, "good_student_all.csv"), index=False)
    df_bad1.to_csv(os.path.join(agg_folder, "bad_student_1st_all.csv"), index=False)
    df_bad_feedback.to_csv(os.path.join(agg_folder, "bad_student_feedback_all.csv"), index=False)

    # Convert to score dictionaries
    score_cols = [col for col in df_good.columns if col not in ["Student", "Patient"]]

    scores_good = {col: df_good[col].tolist() for col in score_cols}
    scores_bad1 = {col: df_bad1[col].tolist() for col in score_cols}
    scores_bad_feedback = {col: df_bad_feedback[col].tolist() for col in score_cols}

    # Generate plots
    print("\n[AGGREGATION] Generating Plot 1: Good vs Bad (1st)...")
    fig1 = plot_scores(
        [scores_good, scores_bad1],
        ["Expert Student", "Novice Student"],
        "Plot 1: Expert Student vs Bad Student (1st Attempt)",
        colors=['skyblue', 'orange']
    )
    fig1.savefig(os.path.join(agg_folder, "plot1_good_vs_bad1.png"), bbox_inches='tight')
    fig1.savefig(os.path.join(agg_folder, "plot1_good_vs_bad1.svg"), bbox_inches='tight')
    plt.close(fig1)

    print("[AGGREGATION] Generating Plot 2: Bad (1st) vs Bad+Feedback...")
    fig2 = plot_scores(
        [scores_bad1, scores_bad_feedback],
        ["Novice Student", "Novice Student + Feedback"],
        "Plot 2: Learning Effect - Before vs After Feedback",
        colors=['orange', 'green']
    )
    fig2.savefig(os.path.join(agg_folder, "plot2_learning_effect.png"), bbox_inches='tight')
    fig2.savefig(os.path.join(agg_folder, "plot2_learning_effect.svg"), bbox_inches='tight')
    plt.close(fig2)

    print("[AGGREGATION] Generating Plot 3: All Three Conditions...")
    fig3 = plot_scores(
        [scores_good, scores_bad1, scores_bad_feedback],
        ["Expert Student", "Novice Student", "Novice Student + Feedback"],
        "Plot 3: Complete Comparison - All Conditions",
        colors=['skyblue', 'orange', 'green']
    )
    fig3.savefig(os.path.join(agg_folder, "plot3_all_conditions.png"), bbox_inches='tight')
    fig3.savefig(os.path.join(agg_folder, "plot3_all_conditions.svg"), bbox_inches='tight')
    plt.close(fig3)

    # ========================================
    # === WILCOXON SIGNED-RANK TESTS ===
    # ========================================

    # Test 1: Good vs Bad (1st)
    print("\n" + "=" * 60)
    print("WILCOXON TEST 1: Expert vs Novice (1st Attempt)")
    print("=" * 60)

    good_sorted = df_good.sort_values("Patient").reset_index(drop=True)
    bad1_sorted = df_bad1.sort_values("Patient").reset_index(drop=True)

    n_pairs_1 = min(len(good_sorted), len(bad1_sorted))
    good_sorted = good_sorted.head(n_pairs_1)
    bad1_sorted = bad1_sorted.head(n_pairs_1)

    print(f"Paired {n_pairs_1} expert/novice sessions for analysis.\n")

    results_1 = []
    for cat in score_cols:
        if cat not in good_sorted.columns or cat not in bad1_sorted.columns:
            continue

        x = good_sorted[cat].astype(float)
        y = bad1_sorted[cat].astype(float)

        valid = (x != y) & (~x.isna()) & (~y.isna())
        x_valid, y_valid = x[valid], y[valid]

        if len(x_valid) < 2:
            print(f"⚠️  Skipping {cat}: not enough valid pairs ({len(x_valid)})")
            continue

        stat, p = wilcoxon(x_valid, y_valid)
        results_1.append({
            "Category": cat,
            "Good Mean": x_valid.mean(),
            "Bad Mean": y_valid.mean(),
            "stat": stat,
            "P-value": p,
            "Significant (<0.05)?": p < 0.05
        })

    if results_1:
        results_df_1 = pd.DataFrame(results_1)
        print("Wilcoxon Signed-Rank Test Results:")
        print(results_df_1[["Category", "stat", "P-value"]].to_string(index=False))
        print("\n")
        print(results_df_1[["Category", "Good Mean", "Bad Mean", "stat", "P-value"]].to_string(index=False))
        results_df_1.to_csv(os.path.join(agg_folder, "wilcoxon_test1_good_vs_bad1.csv"), index=False)
    else:
        print("⚠️ No valid pairs found for Wilcoxon test.")

    # Test 2: Bad (1st) vs Bad (with Feedback)
    print("\n" + "=" * 60)
    print("WILCOXON TEST 2: Novice (1st) vs Novice (with Feedback)")
    print("=" * 60)

    bad1_sorted2 = df_bad1.sort_values("Patient").reset_index(drop=True)
    feedback_sorted = df_bad_feedback.sort_values("Patient").reset_index(drop=True)

    n_pairs_2 = min(len(bad1_sorted2), len(feedback_sorted))
    bad1_sorted2 = bad1_sorted2.head(n_pairs_2)
    feedback_sorted = feedback_sorted.head(n_pairs_2)

    print(f"Paired {n_pairs_2} before/after feedback sessions for analysis.\n")

    results_2 = []
    for cat in score_cols:
        if cat not in bad1_sorted2.columns or cat not in feedback_sorted.columns:
            continue

        x = bad1_sorted2[cat].astype(float)
        y = feedback_sorted[cat].astype(float)

        valid = (x != y) & (~x.isna()) & (~y.isna())
        x_valid, y_valid = x[valid], y[valid]

        if len(x_valid) < 2:
            print(f"⚠️  Skipping {cat}: not enough valid pairs ({len(x_valid)})")
            continue

        stat, p = wilcoxon(x_valid, y_valid)
        results_2.append({
            "Category": cat,
            "Before Mean": x_valid.mean(),
            "After Mean": y_valid.mean(),
            "stat": stat,
            "P-value": p,
            "Significant (<0.05)?": p < 0.05
        })

    if results_2:
        results_df_2 = pd.DataFrame(results_2)
        print("Wilcoxon Signed-Rank Test Results:")
        print(results_df_2[["Category", "stat", "P-value"]].to_string(index=False))
        print("\n")
        print(results_df_2[["Category", "Before Mean", "After Mean", "stat", "P-value"]].to_string(index=False))
        results_df_2.to_csv(os.path.join(agg_folder, "wilcoxon_test2_learning_effect.csv"), index=False)
    else:
        print("⚠️ No valid pairs found for Wilcoxon test.")

    print(f"\n{'=' * 60}")
    print(f"✅ Aggregation complete! Results saved to {agg_folder}")
    print(f"{'=' * 60}")
    print(f"\nGenerated files:")
    print(f"  - Combined CSVs for all 3 conditions")
    print(f"  - plot1_good_vs_bad1.png")
    print(f"  - plot2_learning_effect.png")
    print(f"  - plot3_all_conditions.png")
    print(f"  - wilcoxon_test1_good_vs_bad1.csv")
    print(f"  - wilcoxon_test2_learning_effect.csv")

# =============================
# === Main Entry Point ====
# =============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Testing Pipeline for MI/CBT")

    parser.add_argument("--mode", type=str, choices=["MI", "CBT"], default="MI",
                        help="Therapy mode: MI or CBT")
    parser.add_argument("--start-patient", type=int,
                        help="Start patient index (for parallel execution)")
    parser.add_argument("--end-patient", type=int,
                        help="End patient index (for parallel execution)")
    parser.add_argument("--run-id", type=str, default="exp1",
                        help="Run identifier for grouping parallel runs")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate results from parallel runs")

    args = parser.parse_args()

    if args.aggregate:
        aggregate_results(args)
    else:
        if args.start_patient is None or args.end_patient is None:
            parser.error("--start-patient and --end-patient are required for parallel execution")
        run_parallel_experiments(args)
