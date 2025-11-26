import pandas as pd
from scipy.stats import wilcoxon

# === Load data ===
df = pd.read_csv("Results- First vs Second/student_scores.csv")

# Define the evaluation categories to test
categories = [
    "Acceptance", "Empathy", "Spirit of MI", "Cultivating Change Talk",
    "Softening Sustain Talk", "Partnership", "Open Questions", "Reflections",
    "Affirmations", "Persuade with Permission", "Emphasize Autonomy",
    "Engaging", "Focusing", "Evoking", "Planning", "Summarizing"
]

# Separate first and second tries
first_df = df[df["student"] == "first try"].sort_values("session").reset_index(drop=True)
second_df = df[df["student"] == "second try"].sort_values("session").reset_index(drop=True)

# Ensure matching number of sessions
min_len = min(len(first_df), len(second_df))
first_df = first_df.iloc[:min_len]
second_df = second_df.iloc[:min_len]

results = []

for cat in categories:
    first_vals = first_df[cat].values
    second_vals = second_df[cat].values



    # Wilcoxon test, one-sided: second > first
    stat, p = wilcoxon(first_vals, second_vals, alternative="less")

    results.append({
        "Category": cat,
        "First Try Mean": first_vals.mean(),
        "Second Try Mean": second_vals.mean(),
        "stat": stat,
        "P-value": round(p, 7),
        "Second > First (p < 0.05)?": p < 0.05
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Print results
pd.set_option("display.max_rows", None)
print("\nWilcoxon Signed-Rank Test: Second Try Higher than First Try")
print(results_df[["Category", "stat", "P-value", "Second > First (p < 0.05)?"]].to_string(index=False))
print("\n\n")
print(results_df[["Category","First Try Mean", "Second Try Mean", "stat", "P-value"]].to_string(index=False))
