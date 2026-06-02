import os
import glob
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# === Find the most recent MI_aggregated folder ===
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_repo_root   = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", ".."))
_base_dir    = os.path.join(_repo_root, "saved_tests")
_agg_folders = sorted(glob.glob(os.path.join(_base_dir, "MI_aggregated_*")))
if not _agg_folders:
    raise FileNotFoundError(f"No MI_aggregated_* folders found in {_base_dir}")
latest_folder = _agg_folders[-1]
print(f"Using data from: {latest_folder}\n")

# === Load before/after-feedback CSVs and pair by Patient ===
first_df    = pd.read_csv(os.path.join(latest_folder, "bad_student_1st_all.csv"))
feedback_df = pd.read_csv(os.path.join(latest_folder, "bad_student_feedback_all.csv"))

# Derive category list from columns (everything except metadata)
_exclude = {"Student", "Patient"}
categories = [c for c in first_df.columns if c not in _exclude]

merged = first_df.merge(feedback_df, on="Patient", suffixes=("_first", "_second"))
print(f"Paired {len(merged)} patients for analysis.\n")

results = []

for cat in categories:
    col_first  = f"{cat}_first"
    col_second = f"{cat}_second"
    if col_first not in merged.columns or col_second not in merged.columns:
        continue

    x = pd.to_numeric(merged[col_first],  errors="coerce")  # before feedback
    y = pd.to_numeric(merged[col_second], errors="coerce")  # after feedback

    valid = (~x.isna()) & (~y.isna())
    x_valid, y_valid = x[valid], y[valid]

    if len(x_valid) < 2:
        print(f"Skipping {cat}: not enough valid pairs ({len(x_valid)})")
        continue

    n = len(x_valid)
    # One-sided: H1 = second attempt > first (improvement after feedback), so first < second
    stat, p = wilcoxon(x_valid, y_valid, alternative="less")

    # Rank-biserial r: 1 when all diffs are in expected direction (first < second)
    r = 1 - (2 * stat) / (n * (n + 1))

    results.append({
        "Category":                    cat,
        "Before Mean":                 x_valid.mean(),
        "After Mean":                  y_valid.mean(),
        "n":                           n,
        "stat":                        stat,
        "P-value":                     p,
        "Effect_Size_r":               r,
        "Second > First (p < 0.05)?":  p < 0.05,
    })

if not results:
    print("⚠️ No valid pairs found for Wilcoxon test.")
    raise SystemExit(1)

results_df = pd.DataFrame(results)

# === BH-FDR Multiple Comparisons Correction ===
_reject, _p_fdr, _, _ = multipletests(results_df["P-value"], alpha=0.05, method="fdr_bh")
results_df["p_fdr_bh"]           = list(_p_fdr)
results_df["Significant_fdr_bh"] = _reject

pd.set_option("display.max_rows", None)
print("\nWilcoxon Signed-Rank Test: Learning Effect (Second Try > First Try)")
print(results_df[["Category", "Before Mean", "After Mean", "n", "stat",
                   "P-value", "Effect_Size_r", "p_fdr_bh", "Significant_fdr_bh"]].to_string(index=False))

# === Save Results ===
_out_path = os.path.join(latest_folder, "wilcoxon_test2_learning_effect.csv")
results_df.to_csv(_out_path, index=False)
print(f"\nResults saved to: {_out_path}")
