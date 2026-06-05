import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set once at module level so threads don't race on global rcParams
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 14

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for path in [REPO_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(REPO_ROOT)

import VirtualStudent
from LearningBuddy_Agent import LearningBuddy, MI_MODE, CBT_MODE
from utils.api_key_loader import load_openai_api_key


PATIENT_DIRS = {
    "MI": "Patient_Profiles/MI",
    "CBT": "Patient_Profiles/CBT",
}
STUDENT_CSV_PATHS = {
    "MI": "Student_Profiles/AK_StudentAgent_10Profiles - Prompts for MI Students.csv",
    "CBT": "Student_Profiles/AK_StudentAgent_10Profiles - Prompts for CBT Students.csv",
}
MODE_CONSTANTS = {
    "MI": MI_MODE,
    "CBT": CBT_MODE,
}
VIRTUAL_STUDENT_MODES = {
    "MI": VirtualStudent.VirtualStudent.MI,
    "CBT": VirtualStudent.VirtualStudent.CBT,
}

GOOD_STUDENT_NUM = 11
BAD_STUDENT_NUM = 10
MEDIOCRE_STUDENT_NUM = 12
DEFAULT_PATIENT_COUNT = 20
DEFAULT_SEED = 42
DEFAULT_WORKERS = 20
API_KEY = load_openai_api_key()


def format_duration(seconds):
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"
    if minutes:
        return f"{int(minutes)}m {secs:.1f}s"
    return f"{secs:.1f}s"


def get_patient_files(mode, patient_count, seed=None, random_sample=False):
    patient_dir = PATIENT_DIRS[mode]
    all_files = sorted(
        os.path.join(patient_dir, name)
        for name in os.listdir(patient_dir)
        if os.path.isfile(os.path.join(patient_dir, name))
    )
    if len(all_files) < patient_count:
        raise ValueError(f"Only found {len(all_files)} {mode} patients, need {patient_count}.")
    if random_sample:
        rng = random.Random(seed)
        return rng.sample(all_files, patient_count)
    return all_files[:patient_count]


def score_categories(mode):
    if mode == "MI":
        return [
            "Cultivating Change Talk",
            "Softening Sustain Talk",
            "Partnership",
            "Empathy",
            "Acceptance",
            "Direction",
            "Autonomy Support",
            "Collaboration",
            "Evocation",
        ]
    if mode == "CBT":
        return [
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
            "Homework",
        ]
    raise ValueError(f"Invalid mode: {mode}")


def find_feedback_file(folder_name):
    for filename in ("detailed_feedback.txt", "feedback.txt"):
        path = os.path.join(folder_name, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find feedback file in {folder_name}")


def extract_scores(feedback_path, mode):
    text = Path(feedback_path).read_text(encoding="utf-8")
    scores = {}
    for category in score_categories(mode):
        pattern = re.compile(
            rf"{re.escape(category)}.*?Score:\s*\**\s*(\d+(\.\d+)?)\**",
            re.IGNORECASE | re.DOTALL,
        )
        matches = pattern.findall(text)
        scores[category] = float(matches[-1][0]) if matches else np.nan

    valid_scores = [value for value in scores.values() if not pd.isna(value)]
    scores["Overall Average"] = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else np.nan
    return scores


def extract_recommendations(feedback_path):
    text = Path(feedback_path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"Recommendations\s*(.*?)(?=\n\s*---|\n\s*## Overall Session Score|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else ""


def build_feedback_addon(feedback_text):
    return (
        "Now, reflect on the weaknesses of the previous instructions and strive to enhance your skills "
        "and performance during the therapy session with the patient by strictly incorporating the list "
        "of recommendations. End the conversation after exactly 10 clinician-patient exchanges.\n\n"
        f"# Feedback to incorporate:\n{feedback_text}"
    )


def build_feedback_csv(mode, student_num, feedback_text, feedback_addon, save_as):
    df = pd.read_csv(STUDENT_CSV_PATHS[mode], quotechar='"', skipinitialspace=True)
    df.loc[student_num, "Prompt"] += f"\n\n{feedback_addon}"
    df.to_csv(save_as, index=False)
    return save_as


def save_patient_logprobs(folder_name, turn_logprobs):
    if not turn_logprobs:
        return {
            "turn_logprobs": [],
            "session_avg_logprob": None,
            "session_avg_prob_pct": None,
            "n_turns": 0,
            "logprob_recorded": False,
        }

    session_avg_logprob = sum(turn_logprobs) / len(turn_logprobs)
    summary = {
        "turn_logprobs": turn_logprobs,
        "session_avg_logprob": session_avg_logprob,
        "session_avg_prob_pct": round(math.exp(session_avg_logprob) * 100, 2),
        "n_turns": len(turn_logprobs),
        "logprob_recorded": True,
    }
    with open(os.path.join(folder_name, "patient_logprobs.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def save_student_logprobs(folder_name, student):
    summary = student.get_logprob_summary()
    if summary["n_turns"] > 0:
        with open(os.path.join(folder_name, "student_logprobs.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    return summary


def run_session(mode, patient_file, student_num, patient_number, student_type, prompt_addon=""):
    lb = LearningBuddy(API_KEY or "key", "null", MODE_CONSTANTS[mode])
    student = VirtualStudent.VirtualStudent(
        API_KEY or "key",
        VIRTUAL_STUDENT_MODES[mode],
        student_index=student_num,
        prompt_addon=prompt_addon,
    )

    case = lb.load_patient_profile(patient_file)
    lb.start_session(
        case_study=case,
        session_metadata={
            "patient_number": patient_number,
            "student_type": student_type,
        },
    )

    for _ in range(10):
        current_state = lb.get_session_state()
        interview_messages = current_state.get("interview_messages", [])
        student_utterance = student.get_response(interview_messages)
        lb.send_message(student_utterance)

    lb.end_session()
    final_state = lb.get_session_state()
    folder_name = final_state.get("export_folder") or final_state.get("folder_name", "")
    if not folder_name:
        raise RuntimeError("LearningBuddy did not return a session export folder.")

    feedback_path = find_feedback_file(folder_name)
    logprob_summary = save_patient_logprobs(folder_name, final_state.get("patient_turn_logprobs") or [])
    student_logprob_summary = save_student_logprobs(folder_name, student)
    return {
        "folder_name": folder_name,
        "feedback_path": feedback_path,
        "logprob_summary": logprob_summary,
        "student_logprob_summary": student_logprob_summary,
    }


def make_result_row(mode, condition, student_name, patient_file, session_result):
    slp = session_result["student_logprob_summary"]
    row = {
        "Condition": condition,
        "Student": student_name,
        "Patient": os.path.basename(patient_file),
        "session_folder": session_result["folder_name"],
        "feedback_file": session_result["feedback_path"],
        "patient_avg_logprob": session_result["logprob_summary"]["session_avg_logprob"],
        "patient_avg_prob_pct": session_result["logprob_summary"]["session_avg_prob_pct"],
        "logprob_turns": session_result["logprob_summary"]["n_turns"],
        "logprob_recorded": session_result["logprob_summary"]["logprob_recorded"],
        "student_avg_logprob": slp["session_avg_logprob"],
        "student_avg_prob_pct": slp["session_avg_prob_pct"],
    }
    row.update(extract_scores(session_result["feedback_path"], mode))
    return row


def write_rows(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_dict(df, score_cols):
    return {col: pd.to_numeric(df[col], errors="coerce").dropna().tolist() for col in score_cols}


def plot_scores(score_sets, labels, title, colors=None, ylim=(0, 6.0)):
    if colors is None:
        colors = ["orange", "green", "gold", "skyblue"]

    categories = list(score_sets[0].keys())
    num_sets = len(score_sets)
    averages = []
    std_devs = []

    for scores in score_sets:
        avg = []
        std = []
        for category in categories:
            values = [value for value in scores[category] if not pd.isna(value)]
            avg.append(round(float(np.mean(values)), 2) if values else np.nan)
            std.append(round(float(np.std(values, ddof=1)), 2) if len(values) > 1 else 0)
        averages.append(avg)
        std_devs.append(std)

    x = np.arange(len(categories))
    width = 0.72 / num_sets
    fig_width = max(14, len(categories) * num_sets * 0.5 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars_list = []
    for index, (avg, std, label, color) in enumerate(zip(averages, std_devs, labels, colors)):
        offset = (index - num_sets / 2 + 0.5) * width
        bars = ax.bar(x + offset, avg, width, label=label, color=color, yerr=std, capsize=3)
        bars_list.append(bars)

    ax.set_ylabel("Average Score")
    ax.set_xlabel("Score Category")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(ylim)
    ax.legend(loc="upper left")

    for bars, avg_values in zip(bars_list, averages):
        for bar, height in zip(bars, avg_values):
            if pd.isna(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f"{height:.1f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    return fig


def run_wilcoxon(df_a, df_b, score_cols, label_a, label_b, alternative="two-sided"):
    rows = []
    a_sorted = df_a.sort_values("Patient").reset_index(drop=True)
    b_sorted = df_b.sort_values("Patient").reset_index(drop=True)
    n_pairs = min(len(a_sorted), len(b_sorted))
    a_sorted = a_sorted.head(n_pairs)
    b_sorted = b_sorted.head(n_pairs)

    for category in score_cols:
        x = pd.to_numeric(a_sorted[category], errors="coerce")
        y = pd.to_numeric(b_sorted[category], errors="coerce")
        valid = (~x.isna()) & (~y.isna()) & (x != y)
        x_valid = x[valid]
        y_valid = y[valid]

        row = {
            "Category": category,
            f"{label_a} Mean": x_valid.mean(),
            f"{label_b} Mean": y_valid.mean(),
            "n": len(x_valid),
            "stat": None,
            "P-value": None,
            "Significant (<0.05)?": False,
            "Difference": x_valid.mean() - y_valid.mean(),
        }
        if wilcoxon is not None and len(x_valid) >= 2:
            stat, p_value = wilcoxon(x_valid, y_valid, alternative=alternative)
            row["stat"] = stat
            row["P-value"] = p_value
            row["Significant (<0.05)?"] = p_value < 0.05
        rows.append(row)

    return pd.DataFrame(rows)


def save_mode_outputs(mode, save_folder, condition_rows):
    all_df = pd.concat(
        [pd.DataFrame(rows) for rows in condition_rows.values()],
        ignore_index=True,
    )
    all_df.to_csv(os.path.join(save_folder, "all_conditions.csv"), index=False)

    condition_dfs = {}
    filenames = {
        "good_student": "good_student_all.csv",
        "mediocre_student": "mediocre_student_all.csv",
        "bad_student_1st": "bad_student_1st_all.csv",
        "bad_student_feedback": "bad_student_feedback_all.csv",
    }
    for condition, rows in condition_rows.items():
        df = pd.DataFrame(rows)
        condition_dfs[condition] = df
        df.to_csv(os.path.join(save_folder, filenames[condition]), index=False)

    score_cols = [
        col for col in score_categories(mode) + ["Overall Average"]
        if col in condition_dfs["good_student"].columns
    ]
    patient_logprob_cols = [
        "Condition",
        "Student",
        "Patient",
        "patient_avg_logprob",
        "patient_avg_prob_pct",
        "logprob_turns",
        "logprob_recorded",
        "session_folder",
    ]
    student_logprob_cols = [
        "Condition",
        "Student",
        "Patient",
        "student_avg_logprob",
        "student_avg_prob_pct",
        "session_folder",
    ]
    all_df[patient_logprob_cols].to_csv(os.path.join(save_folder, "patient_logprob_summary.csv"), index=False)
    all_df[student_logprob_cols].to_csv(os.path.join(save_folder, "student_logprob_summary.csv"), index=False)

    plot_specs = [
        (
            "plot1_good_vs_bad1",
            [condition_dfs["good_student"], condition_dfs["bad_student_1st"]],
            ["Expert Student", "Novice Student"],
            ["skyblue", "orange"],
            "Plot 1: Expert Student vs Bad Student (1st Attempt)",
        ),
        (
            "plot2_learning_effect",
            [condition_dfs["bad_student_1st"], condition_dfs["bad_student_feedback"]],
            ["Novice Student", "Novice Student + Feedback"],
            ["orange", "green"],
            "Plot 2: Learning Effect - Before vs After Feedback",
        ),
        (
            "plot3_all_conditions",
            [
                condition_dfs["bad_student_1st"],
                condition_dfs["bad_student_feedback"],
                condition_dfs["mediocre_student"],
                condition_dfs["good_student"],
            ],
            ["Novice Student", "Novice Student + Feedback", "Mediocre Student", "Expert Student"],
            ["orange", "green", "gold", "skyblue"],
            "Plot 3: Complete Comparison - All Conditions",
        ),
    ]

    for stem, dfs, labels, colors, title in plot_specs:
        fig = plot_scores([score_dict(df, score_cols) for df in dfs], labels, title, colors=colors)
        fig.savefig(os.path.join(save_folder, f"{stem}.png"), bbox_inches="tight")
        fig.savefig(os.path.join(save_folder, f"{stem}.svg"), bbox_inches="tight")
        plt.close(fig)

    mean_rows = []
    for category in score_cols:
        row = {"Category": category}
        for condition, df in condition_dfs.items():
            row[f"{condition}_mean"] = pd.to_numeric(df[category], errors="coerce").mean()
            row[f"{condition}_std"] = pd.to_numeric(df[category], errors="coerce").std(ddof=1)
        mean_rows.append(row)
    pd.DataFrame(mean_rows).to_csv(os.path.join(save_folder, "mean_comparison.csv"), index=False)

    wilcoxon_pairs = [
        ("bad1_vs_feedback",    "bad_student_1st",      "bad_student_feedback", "Bad1st",   "Feedback", "less"),
        ("bad1_vs_mediocre",    "bad_student_1st",      "mediocre_student",     "Bad1st",   "Mediocre", "two-sided"),
        ("bad1_vs_good",        "bad_student_1st",      "good_student",         "Bad1st",   "Good",     "two-sided"),
        ("feedback_vs_mediocre","bad_student_feedback", "mediocre_student",     "Feedback", "Mediocre", "two-sided"),
        ("feedback_vs_good",    "bad_student_feedback", "good_student",         "Feedback", "Good",     "two-sided"),
        ("mediocre_vs_good",    "mediocre_student",     "good_student",         "Mediocre", "Good",     "two-sided"),
    ]
    for name, left, right, left_label, right_label, alternative in wilcoxon_pairs:
        result_df = run_wilcoxon(
            condition_dfs[left], condition_dfs[right],
            score_cols, left_label, right_label, alternative=alternative,
        )
        result_df.to_csv(os.path.join(save_folder, f"wilcoxon_{name}.csv"), index=False)

    return {
        "rows": len(all_df),
        "logprob_sessions": int(all_df["logprob_recorded"].sum()),
        "score_columns": score_cols,
    }


def run_mode(mode, parent_folder, patient_count, seed, random_sample):
    mode_start = time.time()
    save_folder = os.path.join(parent_folder, mode)
    os.makedirs(save_folder, exist_ok=True)

    patient_files = get_patient_files(mode, patient_count, seed=seed, random_sample=random_sample)
    student_df = pd.read_csv(STUDENT_CSV_PATHS[mode], quotechar='"', skipinitialspace=True)
    student_names = {
        "good_student": student_df.iloc[GOOD_STUDENT_NUM]["Name"],
        "mediocre_student": student_df.iloc[MEDIOCRE_STUDENT_NUM]["Name"],
        "bad_student_1st": student_df.iloc[BAD_STUDENT_NUM]["Name"],
        "bad_student_feedback": student_df.iloc[BAD_STUDENT_NUM]["Name"] + " (with feedback)",
    }

    metadata = {
        "mode": mode,
        "patient_count": patient_count,
        "seed": seed,
        "random_sample": random_sample,
        "selected_patients": [os.path.basename(path) for path in patient_files],
        "student_indices": {
            "bad": BAD_STUDENT_NUM,
            "mediocre": MEDIOCRE_STUDENT_NUM,
            "good": GOOD_STUDENT_NUM,
            "bad_feedback": BAD_STUDENT_NUM,
        },
        "logprob_expected": True,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(save_folder, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    condition_rows = {
        "good_student": [],
        "mediocre_student": [],
        "bad_student_1st": [],
        "bad_student_feedback": [],
    }

    print(f"\n{'=' * 72}")
    print(f"Starting {mode}: {len(patient_files)} patients x 4 student conditions")
    print(f"Saving to: {save_folder}")
    print(f"{'=' * 72}")

    def _run_patient(patient_idx, patient_file):
        patient_name = os.path.basename(patient_file)
        print(f"\n[{mode}] Patient {patient_idx}/{len(patient_files)}: {patient_name}")
        patient_rows = {}

        # Run good, mediocre, bad_1st in parallel
        with ThreadPoolExecutor(max_workers=3) as inner:
            futs = {
                inner.submit(
                    run_session, mode, patient_file, num, patient_idx,
                    f"{mode.lower()}_{cond}"
                ): cond
                for cond, num in [
                    ("good_student", GOOD_STUDENT_NUM),
                    ("mediocre_student", MEDIOCRE_STUDENT_NUM),
                    ("bad_student_1st", BAD_STUDENT_NUM),
                ]
            }
            for fut in as_completed(futs):
                cond = futs[fut]
                patient_rows[cond] = fut.result()

        # bad_feedback depends on bad_1st result
        bad_feedback_path = patient_rows["bad_student_1st"]["feedback_path"]
        feedback_text = extract_recommendations(bad_feedback_path)
        feedback_addon = build_feedback_addon(feedback_text)
        build_feedback_csv(
            mode, BAD_STUDENT_NUM, feedback_text, feedback_addon,
            os.path.join(save_folder, f"temp_student_feedback_{patient_idx}.csv"),
        )
        patient_rows["bad_student_feedback"] = run_session(
            mode, patient_file, BAD_STUDENT_NUM, patient_idx,
            f"{mode.lower()}_bad_student_feedback",
            prompt_addon=f"\n\n{feedback_addon}",
        )
        return patient_idx, patient_file, patient_rows

    # Run all patients in parallel
    patient_results = [None] * len(patient_files)
    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as executor:
        futures = {
            executor.submit(_run_patient, idx + 1, pf): idx
            for idx, pf in enumerate(patient_files)
        }
        for future in as_completed(futures):
            idx = futures[future]
            patient_results[idx] = future.result()

    # Collect into condition_rows (ordered by original patient order)
    for _, patient_file, patient_rows in patient_results:
        for condition in ("good_student", "mediocre_student", "bad_student_1st", "bad_student_feedback"):
            condition_rows[condition].append(
                make_result_row(mode, condition, student_names[condition], patient_file, patient_rows[condition])
            )

    output_summary = save_mode_outputs(mode, save_folder, condition_rows)
    elapsed = time.time() - mode_start
    with open(os.path.join(save_folder, "elapsed_time.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"Elapsed time: {elapsed:.2f} seconds\n")
        handle.write(f"Elapsed time formatted: {format_duration(elapsed)}\n")

    print(f"\n{mode} complete in {format_duration(elapsed)}")
    print(
        f"Logprob recorded for {output_summary['logprob_sessions']}/{output_summary['rows']} "
        f"{mode} sessions."
    )
    return {
        "mode": mode,
        "save_folder": save_folder,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_time": format_duration(elapsed),
        "sessions": output_summary["rows"],
        "logprob_sessions": output_summary["logprob_sessions"],
    }


def run_all(args):
    overall_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_folder = os.path.join("saved_tests", f"MI_CBT_10_patients_all_students_{timestamp}")
    os.makedirs(parent_folder, exist_ok=True)

    print("Running MI and CBT automated testing")
    print(f"Patient count per mode: {args.patient_count}")
    print("Student conditions: bad, mediocre, good, bad + feedback")
    print(f"Saving combined run to: {parent_folder}")
    print("Logprob capture: enabled via LearningBuddy patient_turn_logprobs and patient_logprobs.json")

    mode_summaries = []
    for offset, mode in enumerate(("MI", "CBT")):
        mode_summaries.append(
            run_mode(mode, parent_folder, args.patient_count, args.seed + offset, args.random_sample)
        )

    total_elapsed = time.time() - overall_start
    run_summary = {
        "patient_count_per_mode": args.patient_count,
        "student_conditions": ["bad_student_1st", "mediocre_student", "good_student", "bad_student_feedback"],
        "total_sessions": sum(summary["sessions"] for summary in mode_summaries),
        "total_logprob_sessions": sum(summary["logprob_sessions"] for summary in mode_summaries),
        "elapsed_seconds": round(total_elapsed, 2),
        "elapsed_time": format_duration(total_elapsed),
        "runs": mode_summaries,
    }
    with open(os.path.join(parent_folder, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
    with open(os.path.join(parent_folder, "elapsed_time.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"Total elapsed time: {total_elapsed:.2f} seconds\n")
        handle.write(f"Total elapsed time formatted: {format_duration(total_elapsed)}\n")

    print(f"\n{'=' * 72}")
    print(f"All MI/CBT runs complete in {format_duration(total_elapsed)}")
    print(
        f"Logprob recorded for {run_summary['total_logprob_sessions']}/"
        f"{run_summary['total_sessions']} sessions."
    )
    print(f"Run summary saved to: {os.path.join(parent_folder, 'run_summary.json')}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MI and CBT patients across bad, mediocre, good, and bad+feedback students."
    )
    parser.add_argument("--patient-count", type=int, default=DEFAULT_PATIENT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample patients instead of taking the first N sorted patient files.",
    )
    run_all(parser.parse_args())
