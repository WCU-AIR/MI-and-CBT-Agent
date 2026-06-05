import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import rankdata, wilcoxon
except ImportError:
    rankdata = None
    wilcoxon = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(REPO_ROOT)

CONDITION_FILES = {
    "bad_student_1st": "bad_student_1st_all.csv",
    "bad_student_feedback": "bad_student_feedback_all.csv",
    "mediocre_student": "mediocre_student_all.csv",
    "good_student": "good_student_all.csv",
}

CONDITION_LABELS = {
    "bad_student_1st": "Bad Student",
    "bad_student_feedback": "Bad Student + Feedback",
    "mediocre_student": "Mediocre Student",
    "good_student": "Good Student",
}

CONDITION_COLORS = {
    "bad_student_1st": "orange",
    "bad_student_feedback": "green",
    "mediocre_student": "gold",
    "good_student": "skyblue",
}

EXCLUDED_COLUMNS = {
    "Condition",
    "Student",
    "Patient",
    "session_folder",
    "feedback_file",
    "patient_avg_logprob",
    "patient_avg_prob_pct",
    "logprob_turns",
    "logprob_recorded",
    "student_avg_logprob",
    "student_avg_prob_pct",
}

COMPARISONS = [
    ("bad1_vs_feedback", "bad_student_1st", "bad_student_feedback", "less"),
    ("bad1_vs_mediocre", "bad_student_1st", "mediocre_student", "two-sided"),
    ("bad1_vs_good", "bad_student_1st", "good_student", "two-sided"),
    ("feedback_vs_mediocre", "bad_student_feedback", "mediocre_student", "two-sided"),
    ("feedback_vs_good", "bad_student_feedback", "good_student", "two-sided"),
    ("mediocre_vs_good", "mediocre_student", "good_student", "two-sided"),
]

PLOT_SPECS = [
    (
        "plot1_bad_vs_good",
        ["bad_student_1st", "good_student"],
        "Plot 1: Bad vs Good Students",
    ),
    (
        "plot2_learning_effect",
        ["bad_student_1st", "bad_student_feedback"],
        "Plot 2: Learning Effect - Before vs After Feedback",
    ),
    (
        "plot3_all_conditions",
        ["bad_student_1st", "bad_student_feedback", "mediocre_student", "good_student"],
        "Plot 3: Complete Comparison - All Conditions",
    ),
    (
        "plot4_bad_mediocre_good",
        ["bad_student_1st", "mediocre_student", "good_student"],
        "Plot 4: Bad vs Mediocre vs Good Students",
    ),
]


def latest_run_folder():
    folders = [
        path for path in glob(os.path.join("saved_tests", "MI_CBT_all_patients_all_students_*"))
        if os.path.isdir(path)
    ]
    if not folders:
        raise FileNotFoundError("No saved_tests/MI_CBT_all_patients_all_students_* folders found.")
    return max(folders, key=os.path.getmtime)


def mode_from_folder(mode_folder):
    metadata_path = os.path.join(mode_folder, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("mode", os.path.basename(mode_folder))
    return os.path.basename(mode_folder)


def load_condition_data(mode_folder):
    frames = {}
    for condition, filename in CONDITION_FILES.items():
        path = os.path.join(mode_folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required condition CSV: {path}")
        frames[condition] = pd.read_csv(path)
    return frames


def get_score_columns(frames):
    common = set.intersection(*(set(df.columns) for df in frames.values()))
    score_cols = []
    for col in frames["bad_student_1st"].columns:
        if col in common and col not in EXCLUDED_COLUMNS:
            converted = pd.to_numeric(frames["bad_student_1st"][col], errors="coerce")
            if converted.notna().any():
                score_cols.append(col)
    return score_cols


def benjamini_hochberg(p_values):
    indexed = [(idx, p) for idx, p in enumerate(p_values) if not pd.isna(p)]
    adjusted = [np.nan] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running_min = 1.0
    for rank_from_end, (idx, p) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running_min = min(running_min, p * m / rank)
        adjusted[idx] = min(running_min, 1.0)
    return adjusted


def paired_rank_biserial(x, y):
    if rankdata is None:
        return np.nan
    diff = y - x
    non_zero = diff[diff != 0]
    if len(non_zero) == 0:
        return np.nan
    ranks = rankdata(np.abs(non_zero))
    positive = ranks[non_zero > 0].sum()
    negative = ranks[non_zero < 0].sum()
    return (positive - negative) / ranks.sum()


def run_paired_stats(frames, score_cols, comparison_name, left, right, alternative):
    left_df = frames[left]
    right_df = frames[right]
    merged = left_df.merge(
        right_df,
        on="Patient",
        suffixes=("_left", "_right"),
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for col in score_cols:
        x = pd.to_numeric(merged[f"{col}_left"], errors="coerce")
        y = pd.to_numeric(merged[f"{col}_right"], errors="coerce")
        valid = ~(x.isna() | y.isna())
        x_valid = x[valid]
        y_valid = y[valid]
        diff = y_valid - x_valid
        non_tie = diff != 0
        x_test = x_valid[non_tie]
        y_test = y_valid[non_tie]

        stat = np.nan
        p_value = np.nan
        if wilcoxon is not None and len(x_test) >= 2:
            stat, p_value = wilcoxon(x_test, y_test, alternative=alternative)

        rows.append({
            "Comparison": comparison_name,
            "Left Condition": left,
            "Right Condition": right,
            "Category": col,
            "Left Mean": x_valid.mean(),
            "Right Mean": y_valid.mean(),
            "Mean Difference (Right-Left)": diff.mean(),
            "n_pairs_available": len(merged),
            "n_valid_pairs": len(x_valid),
            "n_ties_removed_for_wilcoxon": int((diff == 0).sum()),
            "n_used_in_wilcoxon": len(x_test),
            "stat": stat,
            "P-value": p_value,
            "Effect_Size_rank_biserial": paired_rank_biserial(x_valid.to_numpy(), y_valid.to_numpy()),
        })

    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = benjamini_hochberg(result["P-value"].tolist())
    result["Significant_raw_0.05"] = result["P-value"] < 0.05
    result["Significant_fdr_bh_0.05"] = result["p_fdr_bh"] < 0.05
    return result


def summarize_conditions(frames, score_cols):
    rows = []
    for condition, df in frames.items():
        for col in score_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            rows.append({
                "Condition": condition,
                "Category": col,
                "n_rows": len(df),
                "n_valid": int(values.notna().sum()),
                "n_missing": int(values.isna().sum()),
                "Mean": values.mean(),
                "Std": values.std(ddof=1),
                "SEM": values.std(ddof=1) / math.sqrt(values.notna().sum()) if values.notna().sum() > 1 else np.nan,
                "Median": values.median(),
                "Min": values.min(),
                "Max": values.max(),
            })
    return pd.DataFrame(rows)


def diagnostics(frames, score_cols):
    rows = []
    all_patients = set()
    for df in frames.values():
        all_patients.update(df["Patient"].dropna().tolist())
    for condition, df in frames.items():
        duplicate_count = int(df["Patient"].duplicated().sum())
        missing_patients = sorted(all_patients - set(df["Patient"].dropna().tolist()))
        rows.append({
            "Condition": condition,
            "Rows": len(df),
            "Unique Patients": df["Patient"].nunique(),
            "Duplicate Patient Rows": duplicate_count,
            "Patients Missing From This Condition": len(missing_patients),
            "Missing Patient Names": "; ".join(missing_patients[:20]),
        })

    for comparison_name, left, right, _alternative in COMPARISONS:
        merged = frames[left][["Patient"]].merge(frames[right][["Patient"]], on="Patient", how="inner")
        rows.append({
            "Condition": comparison_name,
            "Rows": len(merged),
            "Unique Patients": merged["Patient"].nunique(),
            "Duplicate Patient Rows": int(merged["Patient"].duplicated().sum()),
            "Patients Missing From This Condition": np.nan,
            "Missing Patient Names": "paired comparison rows",
        })
    return pd.DataFrame(rows)


def score_dict(df, score_cols):
    return {
        col: pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        for col in score_cols
    }


def plot_scores(frames, conditions, score_cols, title, output_stem):
    categories = score_cols
    x = np.arange(len(categories))
    width = 0.72 / len(conditions)
    fig_width = max(14, len(categories) * len(conditions) * 0.5 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for idx, condition in enumerate(conditions):
        scores = score_dict(frames[condition], score_cols)
        means = [np.mean(scores[col]) if scores[col] else np.nan for col in categories]
        sems = [
            np.std(scores[col], ddof=1) / math.sqrt(len(scores[col])) if len(scores[col]) > 1 else 0
            for col in categories
        ]
        offset = (idx - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=sems,
            capsize=3,
            label=CONDITION_LABELS[condition],
            color=CONDITION_COLORS[condition],
        )
        for bar, value in zip(bars, means):
            if pd.isna(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value / 2,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_ylabel("Average Score")
    ax.set_xlabel("Score Category")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 6.0)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(f"{output_stem}.png", bbox_inches="tight")
    fig.savefig(f"{output_stem}.svg", bbox_inches="tight")
    plt.close(fig)


def write_mode_outputs(mode_folder, output_folder):
    mode = mode_from_folder(mode_folder)
    frames = load_condition_data(mode_folder)
    score_cols = get_score_columns(frames)
    if not score_cols:
        raise ValueError(f"No score columns found in {mode_folder}")

    os.makedirs(output_folder, exist_ok=True)
    diagnostics_df = diagnostics(frames, score_cols)
    diagnostics_df.to_csv(os.path.join(output_folder, "diagnostics_row_counts.csv"), index=False)

    summary_df = summarize_conditions(frames, score_cols)
    summary_df.to_csv(os.path.join(output_folder, "condition_summary_stats.csv"), index=False)

    wide_summary = summary_df.pivot(index="Category", columns="Condition", values="Mean")
    wide_summary.to_csv(os.path.join(output_folder, "condition_means_wide.csv"))

    all_stats = []
    for comparison_name, left, right, alternative in COMPARISONS:
        result = run_paired_stats(frames, score_cols, comparison_name, left, right, alternative)
        result.to_csv(os.path.join(output_folder, f"wilcoxon_{comparison_name}.csv"), index=False)
        all_stats.append(result)
    pd.concat(all_stats, ignore_index=True).to_csv(
        os.path.join(output_folder, "wilcoxon_all_comparisons.csv"),
        index=False,
    )

    for stem, conditions, title in PLOT_SPECS:
        plot_scores(
            frames,
            conditions,
            score_cols,
            f"{mode} {title}",
            os.path.join(output_folder, stem),
        )

    readme = [
        f"Recomputed analysis for {mode}",
        f"Source folder: {mode_folder}",
        f"Score columns: {', '.join(score_cols)}",
        "",
        "Wilcoxon tests are paired by exact Patient name using an inner merge.",
        "Rows with equal paired values are counted as ties and excluded from the Wilcoxon statistic,",
        "because zero-difference pairs do not contribute to the signed-rank test.",
    ]
    Path(os.path.join(output_folder, "README_recomputed.txt")).write_text("\n".join(readme), encoding="utf-8")

    return {
        "mode": mode,
        "source_folder": mode_folder,
        "output_folder": output_folder,
        "score_columns": score_cols,
        "condition_rows": {
            condition: len(df) for condition, df in frames.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recompute plots and statistical tables from an existing MI/CBT all-patients run."
    )
    parser.add_argument(
        "--run-folder",
        default=None,
        help="Existing saved_tests/MI_CBT_all_patients_all_students_* folder. Defaults to latest.",
    )
    args = parser.parse_args()

    run_folder = args.run_folder or latest_run_folder()
    if not os.path.isdir(run_folder):
        raise FileNotFoundError(f"Run folder does not exist: {run_folder}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(run_folder, f"recomputed_analysis_{timestamp}")
    os.makedirs(output_root, exist_ok=True)

    summaries = []
    for mode in ("MI", "CBT"):
        mode_folder = os.path.join(run_folder, mode)
        if os.path.isdir(mode_folder):
            summaries.append(write_mode_outputs(mode_folder, os.path.join(output_root, mode)))

    with open(os.path.join(output_root, "recompute_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "source_run_folder": run_folder,
            "output_folder": output_root,
            "modes": summaries,
        }, handle, indent=2)

    print(f"Recomputed analysis saved to: {output_root}")
    for summary in summaries:
        print(f"{summary['mode']}: {summary['condition_rows']}")


if __name__ == "__main__":
    main()
