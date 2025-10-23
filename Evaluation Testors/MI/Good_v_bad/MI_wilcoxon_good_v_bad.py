import pandas as pd
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
        "Second > First (p < 0.05)?": p < 0.05
    })

# === Summarize Results ===
if results:
    results_df = pd.DataFrame(results)
    results_df["Significant (<0.05)?"] = results_df["P-value"] < 0.05
    print("Wilcoxon Signed-Rank Test Results:")
    print(results_df[["Category", "stat", "P-value"]].to_string(index=False))
    print("\n\n")
    print(results_df[["Category","Good Mean", "Bad Mean", "stat", "P-value"]].to_string(index=False))
else:
    print("⚠️ No valid pairs found for Wilcoxon test.")