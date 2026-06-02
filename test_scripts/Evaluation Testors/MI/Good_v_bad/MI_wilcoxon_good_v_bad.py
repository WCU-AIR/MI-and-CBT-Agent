import os
import glob
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# === Find the most recent MI_aggregated folder ===
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root  = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", ".."))
_base_dir   = os.path.join(_repo_root, "saved_tests")
_agg_folders = sorted(glob.glob(os.path.join(_base_dir, "MI_aggregated_*")))
if not _agg_folders:
    raise FileNotFoundError(f"No MI_aggregated_* folders found in {_base_dir}")
latest_folder = _agg_folders[-1]
print(f"Using data from: {latest_folder}\n")

# === Load separate good / bad CSVs and pair by Patient ===
good_df = pd.read_csv(os.path.join(latest_folder, "good_student_all.csv"))
bad_df  = pd.read_csv(os.path.join(latest_folder, "bad_student_1st_all.csv"))

# Derive category list from columns (everything except metadata)
_exclude = {"Student", "Patient"}
categories = [c for c in good_df.columns if c not in _exclude]

merged = good_df.merge(bad_df, on="Patient", suffixes=("_good", "_bad"))
print(f"Paired {len(merged)} patients for analysis.\n")

results = []

# === Perform Wilcoxon Signed-Rank Tests ===
for cat in categories:
    col_good = f"{cat}_good"
    col_bad  = f"{cat}_bad"
    if col_good not in merged.columns or col_bad not in merged.columns:
        continue

    x = pd.to_numeric(merged[col_good], errors="coerce")
    y = pd.to_numeric(merged[col_bad],  errors="coerce")

    # Drop NaNs and identical pairs (Wilcoxon requires at least one non-zero diff)
    valid = (x != y) & (~x.isna()) & (~y.isna())
    x_valid, y_valid = x[valid], y[valid]

    if len(x_valid) < 2:
        continue

    n = len(x_valid)
    stat, p = wilcoxon(x_valid, y_valid)

    # Rank-biserial r: 1 = all diffs in same direction, 0 = no effect
    r = 1 - (2 * stat) / (n * (n + 1))

    results.append({
        "Category":      cat,
        "Good Mean":     x_valid.mean(),
        "Bad Mean":      y_valid.mean(),
        "n":             n,
        "stat":          stat,
        "P-value":       p,
        "Effect_Size_r": r,
    })

if not results:
    print("⚠️ No valid pairs found for Wilcoxon test.")
    raise SystemExit(1)

results_df = pd.DataFrame(results)

# === BH-FDR Multiple Comparisons Correction ===
_reject, _p_fdr, _, _ = multipletests(results_df["P-value"], alpha=0.05, method="fdr_bh")
results_df["p_fdr_bh"]           = list(_p_fdr)
results_df["Significant_fdr_bh"] = _reject

# Columns kept for plot_significance_summary and backward-compat readers
results_df["Good > Bad (p < 0.05)?"] = results_df["P-value"] < 0.05
results_df["Significant (<0.05)?"]   = results_df["Good > Bad (p < 0.05)?"]
results_df["Difference"]  = results_df["Good Mean"] - results_df["Bad Mean"]
results_df["P_Value"]     = results_df["P-value"]
results_df["Significance"] = results_df["P-value"].apply(
    lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
)

print("Wilcoxon Signed-Rank Test Results (Good vs Bad):")
print(results_df[["Category", "Good Mean", "Bad Mean", "n", "stat",
                   "P-value", "Effect_Size_r", "p_fdr_bh", "Significant_fdr_bh"]].to_string(index=False))

# === Save Results ===
_out_path = os.path.join(latest_folder, "wilcoxon_test1_good_vs_bad1.csv")
results_df.to_csv(_out_path, index=False)
print(f"\nResults saved to: {_out_path}")


def plot_significance_summary(wilcoxon_df):
    """Create a summary plot showing which categories have significant differences."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 10

    # Sort by difference (largest to smallest)
    df_sorted = wilcoxon_df.sort_values('Difference', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    categories = df_sorted['Category'].values
    differences = df_sorted['Difference'].values
    p_values = df_sorted['P_Value'].values

    # Color bars based on significance
    colors = []
    for p in p_values:
        if pd.isna(p):
            colors.append('gray')
        elif p < 0.001:
            colors.append('darkgreen')
        elif p < 0.01:
            colors.append('green')
        elif p < 0.05:
            colors.append('lightgreen')
        else:
            colors.append('lightcoral')

    bars = ax.barh(categories, differences, color=colors)

    # Add p-value annotations
    for i, (bar, p, sig) in enumerate(zip(bars, p_values, df_sorted['Significance'].values)):
        x_pos = bar.get_width()
        if x_pos > 0:
            ha = 'left'
            x_pos += 0.02
        else:
            ha = 'right'
            x_pos -= 0.02

        if not pd.isna(p):
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f'{sig}', ha=ha, va='center', fontweight='bold')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Mean Difference (Expert - Novice)')
    ax.set_title(
        'Statistical Significance of Performance Differences\n*** p<0.001, ** p<0.01, * p<0.05, ns = not significant')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig