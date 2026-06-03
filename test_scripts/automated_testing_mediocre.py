import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from glob import glob

import pandas as pd

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for path in [REPO_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from automated_testing import (
    extract_scores,
    get_patient_files,
    plot_scores,
    run_test,
)


MEDIOCRE_STUDENT_NUM = 12
SAMPLE_SIZE = 10
DEFAULT_SEED = 42
MODES_TO_RUN = ("MI", "CBT")


def format_duration(seconds):
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"
    if minutes:
        return f"{int(minutes)}m {secs:.1f}s"
    return f"{secs:.1f}s"


def student_csv_path(mode):
    if mode == "MI":
        return "Student_Profiles/AK_StudentAgent_10Profiles - Prompts for MI Students.csv"
    return "Student_Profiles/AK_StudentAgent_10Profiles - Prompts for CBT Students.csv"


def latest_aggregate_folder(mode):
    pattern = os.path.join("saved_tests", f"{mode}_aggregated_*")
    folders = [folder for folder in glob(pattern) if os.path.isdir(folder)]
    if not folders:
        raise FileNotFoundError(f"No existing aggregate folders found for {mode}: {pattern}")
    return max(folders, key=os.path.getmtime)


def find_feedback_file(folder_name, label):
    possible_names = [
        f"detailed_feedback_{label}.txt",
        "detailed_feedback.txt",
        "feedback.txt",
    ]

    for name in possible_names:
        path = os.path.join(folder_name, name)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find feedback file in {folder_name}. "
        f"Tried: {', '.join(possible_names)}"
    )


def write_rows(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_mediocre_sample(mode, patient_files, save_folder):
    student_df = pd.read_csv(student_csv_path(mode), quotechar='"', skipinitialspace=True)
    mediocre_name = student_df.iloc[MEDIOCRE_STUDENT_NUM]["Name"]

    rows = []
    feedback_files = []

    for patient_idx, patient_file in enumerate(patient_files, start=1):
        patient_name = os.path.basename(patient_file)
        print(f"\n[MEDIOCRE] Running patient {patient_idx}/{len(patient_files)}: {patient_name}")

        results = run_test(
            mode,
            patient_file,
            MEDIOCRE_STUDENT_NUM,
            patient_number=patient_idx,
            student_type="mediocre_student",
        )
        feedback_path = find_feedback_file(results["folder_name"], "mediocre_student")
        feedback_files.append(feedback_path)

        scores = extract_scores([feedback_path], mode)
        row = {
            "Student": mediocre_name,
            "Patient": patient_name,
        }
        for key, values in scores.items():
            row[key] = values[0] if values else None
        rows.append(row)

    score_cols = [key for key in rows[0].keys() if key not in ["Student", "Patient"]]
    write_rows(
        os.path.join(save_folder, "mediocre_student_scores.csv"),
        rows,
        ["Student", "Patient"] + score_cols,
    )

    return pd.DataFrame(rows), feedback_files


def load_existing_samples(aggregate_folder, sample_size, seed):
    good_path = os.path.join(aggregate_folder, "good_student_all.csv")
    bad_path = os.path.join(aggregate_folder, "bad_student_1st_all.csv")

    if not os.path.exists(good_path):
        raise FileNotFoundError(f"Missing expert results CSV: {good_path}")
    if not os.path.exists(bad_path):
        raise FileNotFoundError(f"Missing bad-student results CSV: {bad_path}")

    expert_df = pd.read_csv(good_path)
    bad_df = pd.read_csv(bad_path)

    if len(expert_df) < sample_size:
        raise ValueError(f"Expert results only have {len(expert_df)} rows, need {sample_size}")
    if len(bad_df) < sample_size:
        raise ValueError(f"Bad-student results only have {len(bad_df)} rows, need {sample_size}")

    expert_sample = expert_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    bad_sample = bad_df.sample(n=sample_size, random_state=seed + 1).reset_index(drop=True)
    return expert_sample, bad_sample


def score_dict(df):
    score_cols = [col for col in df.columns if col not in ["Student", "Patient"]]
    return {col: pd.to_numeric(df[col], errors="coerce").tolist() for col in score_cols}


def compare_groups(mediocre_df, expert_df, bad_df, save_folder):
    score_cols = [col for col in mediocre_df.columns if col not in ["Student", "Patient"]]

    tagged = pd.concat(
        [
            mediocre_df.assign(Condition="Mediocre Student"),
            expert_df.assign(Condition="Existing Expert Sample"),
            bad_df.assign(Condition="Existing Bad Sample"),
        ],
        ignore_index=True,
    )
    tagged.to_csv(os.path.join(save_folder, "comparison_all.csv"), index=False)

    summary_rows = []
    stats_rows = []

    for col in score_cols:
        mediocre_values = pd.to_numeric(mediocre_df[col], errors="coerce").dropna()
        expert_values = pd.to_numeric(expert_df[col], errors="coerce").dropna()
        bad_values = pd.to_numeric(bad_df[col], errors="coerce").dropna()

        summary_rows.append(
            {
                "Category": col,
                "Mediocre Mean": mediocre_values.mean(),
                "Expert Sample Mean": expert_values.mean(),
                "Bad Sample Mean": bad_values.mean(),
                "Mediocre Std": mediocre_values.std(ddof=1),
                "Expert Sample Std": expert_values.std(ddof=1),
                "Bad Sample Std": bad_values.std(ddof=1),
            }
        )

        if mannwhitneyu is not None:
            for comparison_name, comparison_values in [
                ("mediocre_vs_expert", expert_values),
                ("mediocre_vs_bad", bad_values),
            ]:
                if len(mediocre_values) < 2 or len(comparison_values) < 2:
                    continue
                stat, p_value = mannwhitneyu(
                    mediocre_values,
                    comparison_values,
                    alternative="two-sided",
                )
                stats_rows.append(
                    {
                        "Comparison": comparison_name,
                        "Category": col,
                        "Mediocre Mean": mediocre_values.mean(),
                        "Comparison Mean": comparison_values.mean(),
                        "U-statistic": stat,
                        "P-value": p_value,
                        "Significant (<0.05)?": p_value < 0.05,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    stats_df = pd.DataFrame(stats_rows)

    summary_df.to_csv(os.path.join(save_folder, "mean_comparison.csv"), index=False)
    if mannwhitneyu is not None:
        stats_df.to_csv(os.path.join(save_folder, "mannwhitney_comparison.csv"), index=False)

    fig = plot_scores(
        [score_dict(bad_df), score_dict(mediocre_df), score_dict(expert_df)],
        ["Existing Bad Sample", "Mediocre Student", "Existing Expert Sample"],
        "Mediocre Student vs Existing Bad and Expert Samples",
        colors=["orange", "green", "skyblue"],
    )
    fig.savefig(os.path.join(save_folder, "plot_mediocre_vs_existing.png"), bbox_inches="tight")
    fig.savefig(os.path.join(save_folder, "plot_mediocre_vs_existing.svg"), bbox_inches="tight")

    return summary_df, stats_df


def run_mode(mode, seed, parent_folder):
    start_time = time.time()
    rng = random.Random(seed)

    all_patients = get_patient_files(mode)
    if len(all_patients) < SAMPLE_SIZE:
        raise ValueError(
            f"Only found {len(all_patients)} {mode} patient profiles, "
            f"need {SAMPLE_SIZE}"
        )

    patient_files = rng.sample(all_patients, SAMPLE_SIZE)
    aggregate_folder = latest_aggregate_folder(mode)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(parent_folder, mode)
    os.makedirs(save_folder, exist_ok=True)

    metadata = {
        "mode": mode,
        "sample_size": SAMPLE_SIZE,
        "seed": seed,
        "aggregate_folder": aggregate_folder,
        "selected_patients": [os.path.basename(path) for path in patient_files],
        "timestamp": timestamp,
        "mediocre_student_index": MEDIOCRE_STUDENT_NUM,
    }
    with open(os.path.join(save_folder, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\n{'=' * 72}")
    print(f"Starting {mode} mediocre-student run")
    print(f"{'=' * 72}")
    print(f"Seed: {seed}")
    print(f"Selected {len(patient_files)} random patients")
    print(f"Existing comparison source: {aggregate_folder}")
    print(f"Saving to: {save_folder}")

    mediocre_df, feedback_files = run_mediocre_sample(mode, patient_files, save_folder)
    expert_df, bad_df = load_existing_samples(aggregate_folder, SAMPLE_SIZE, seed)

    expert_df.to_csv(os.path.join(save_folder, "expert_existing_sample.csv"), index=False)
    bad_df.to_csv(os.path.join(save_folder, "bad_existing_sample.csv"), index=False)

    summary_df, stats_df = compare_groups(mediocre_df, expert_df, bad_df, save_folder)

    elapsed_seconds = time.time() - start_time
    with open(os.path.join(save_folder, "elapsed_time.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"Elapsed time: {elapsed_seconds:.2f} seconds\n")
        handle.write(f"Elapsed time formatted: {format_duration(elapsed_seconds)}\n")

    print("\nMean comparison:")
    print(summary_df.to_string(index=False))
    if mannwhitneyu is None:
        print("\nSciPy is not installed; skipped Mann-Whitney U comparison.")
    else:
        print("\nMann-Whitney U comparison:")
        print(stats_df.to_string(index=False))
    print(f"\n{mode} complete in {format_duration(elapsed_seconds)}")
    print(f"Results saved to {save_folder}")
    print(f"Feedback files collected: {len(feedback_files)}")

    return {
        "mode": mode,
        "seed": seed,
        "sample_size": SAMPLE_SIZE,
        "save_folder": save_folder,
        "aggregate_folder": aggregate_folder,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "elapsed_time": format_duration(elapsed_seconds),
        "feedback_files": len(feedback_files),
    }


def run_all():
    overall_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_folder = os.path.join("saved_tests", f"mediocre_both_modes_{timestamp}")
    os.makedirs(parent_folder, exist_ok=True)

    print("Running mediocre-student automated testing")
    print(f"Modes: {', '.join(MODES_TO_RUN)}")
    print(f"Sample size per mode: {SAMPLE_SIZE}")
    print(f"Saving combined run to: {parent_folder}")

    summaries = []
    for offset, mode in enumerate(MODES_TO_RUN):
        summaries.append(run_mode(mode, DEFAULT_SEED + offset, parent_folder))

    total_elapsed = time.time() - overall_start
    run_summary = {
        "sample_size_per_mode": SAMPLE_SIZE,
        "base_seed": DEFAULT_SEED,
        "modes": list(MODES_TO_RUN),
        "elapsed_seconds": round(total_elapsed, 2),
        "elapsed_time": format_duration(total_elapsed),
        "runs": summaries,
    }

    with open(os.path.join(parent_folder, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    with open(os.path.join(parent_folder, "elapsed_time.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"Total elapsed time: {total_elapsed:.2f} seconds\n")
        handle.write(f"Total elapsed time formatted: {format_duration(total_elapsed)}\n")

    print(f"\n{'=' * 72}")
    print(f"All mediocre-student runs complete in {format_duration(total_elapsed)}")
    print(f"Run summary saved to {os.path.join(parent_folder, 'run_summary.json')}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    run_all()
