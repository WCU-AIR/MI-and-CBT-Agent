import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_scores(good_student, bad_student, title="Evaluation Scores for CBT"):
    """
    Plots grouped bar chart comparing two students’ average CBT scores
    with error bars showing ±SEM (standard error of the mean).
    """
    categories = list(good_student.keys())

    # Compute mean and SEM (standard error of the mean)
    avg_set1 = [np.mean(good_student[cat]) for cat in categories]
    sem_set1 = [np.std(good_student[cat], ddof=1) / np.sqrt(len(good_student[cat])) 
                if len(good_student[cat]) > 1 else 0 for cat in categories]

    avg_set2 = [np.mean(bad_student[cat]) for cat in categories]
    sem_set2 = [np.std(bad_student[cat], ddof=1) / np.sqrt(len(bad_student[cat])) 
                if len(bad_student[cat]) > 1 else 0 for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars with error bars (±SEM)
    bars1 = ax.bar(x - width / 2, avg_set1, width, 
                   yerr=sem_set1, label='Good Student', color='skyblue',
                   capsize=5, ecolor='gray', alpha=0.9)
    bars2 = ax.bar(x + width / 2, avg_set2, width, 
                   yerr=sem_set2, label='Bad Student', color='orange',
                   capsize=5, ecolor='gray', alpha=0.9)

    # Labels and formatting
    ax.set_ylabel('Average Score')
    ax.set_xlabel('Score Category')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylim(0, 6.5)
    ax.legend()

    # Add text labels above bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


# =====================
# READ AND PREPARE DATA
# =====================
# df = pd.read_csv("saved_tests/run_20251015_104140/CBT_scores.csv")
#
# # Extract unique students
# students = df['Student'].unique()
# print("Students found:", students)
#
# # Example: compare expert vs novice
# good_student_name = "12_Expert_CBTVirtuoso"
# bad_student_name = "11_Novice_UndirectedAdviceGiver"
#
# # Select their data
# good_df = df[df["Student"] == good_student_name]
# bad_df = df[df["Student"] == bad_student_name]
#
# # Extract numeric columns (categories only)
# score_cols = [c for c in df.columns if c not in ["Student", "Patient", "Overall Average"]]
#
# # Build category dictionaries
# good_student = {col: good_df[col].dropna().tolist() for col in score_cols}
# bad_student  = {col: bad_df[col].dropna().tolist()  for col in score_cols}
#
# # =====================
# # PLOT
# # =====================
# fig = plot_scores(good_student, bad_student)
# plt.show()
