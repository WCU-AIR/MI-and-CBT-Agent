import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon

# === Load CSV ===
df = pd.read_csv("Results- Good vs Bad/extracted_scores.csv")

# Define your evaluation categories
categories = [
    "Acceptance", "Empathy", "Spirit of MI", "Cultivating Change Talk",
    "Softening Sustain Talk", "Partnership", "Open Questions", "Reflections",
    "Affirmations", "Persuade with Permission", "Emphasize Autonomy",
    "Engaging", "Focusing", "Evoking", "Planning", "Summarizing"
]

# Split into good and bad groups
good_df = df[df["Student"] == "good student"].sort_values("session_folder").reset_index(drop=True)
bad_df  = df[df["Student"] == "bad student"].sort_values("session_folder").reset_index(drop=True)

# Ensure same number of rows
n_pairs = min(len(good_df), len(bad_df))
good_df = good_df.head(n_pairs)
bad_df  = bad_df.head(n_pairs)

print(f"Paired {n_pairs} good/bad sessions for analysis.\n")

results = []

# === Perform Wilcoxon Signed-Rank Tests ===
for cat in categories:
    if cat not in good_df.columns or cat not in bad_df.columns:
        continue

    x = good_df[cat].astype(float)
    y = bad_df[cat].astype(float)

    # Drop NaNs and identical pairs
    valid = (x != y) & (~x.isna()) & (~y.isna())
    x_valid, y_valid = x[valid], y[valid]

    if len(x_valid) < 2:
        continue  # skip if not enough pairs for test

    stat, p = wilcoxon(x_valid, y_valid)
    results.append({
        "Category": cat,
        "Good Mean": x_valid.mean(),
        "Bad Mean": y_valid.mean(),
        "stat": stat,
        "P-value": round(p, 7),
        "Good > Bad (p < 0.05)?": p < 0.05
    })

# === Summarize Results ===
if results:
    results_df = pd.DataFrame(results)
    results_df["Significant (<0.05)?"] = results_df["P-value"] < 0.05
    results_df["Difference"] = results_df["Good Mean"] - results_df["Bad Mean"]
    results_df["P_Value"] = results_df["P-value"]
    results_df["Significance"] = results_df["P-value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    )
    print("Wilcoxon Signed-Rank Test Results:")
    print(results_df[["Category", "stat", "P-value"]].to_string(index=False))
    print("\n\n")
    print(results_df[["Category","Good Mean", "Bad Mean", "stat", "P-value"]].to_string(index=False))
else:
    print("⚠️ No valid pairs found for Wilcoxon test.")


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