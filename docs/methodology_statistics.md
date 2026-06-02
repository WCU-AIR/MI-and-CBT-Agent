# Evaluation Methodology and Statistical Analysis

This document describes the statistical methodology used in the hypothesis-testing
scripts for the MI (Motivational Interviewing) evaluation component of the
LearningBuddy framework.

Scripts documented here:

- `test_scripts/Evaluation Testors/MI/Good_v_bad/MI_wilcoxon_good_v_bad.py`
- `test_scripts/Evaluation Testors/MI/First_v_second/MI_wilcoxon_first_v_second.py`

---

## 1. Evaluation Rubric

Each simulated counselling session is scored by the Evaluator Agent on nine
Motivational Interviewing (MI) criteria using a 5-point Likert scale:

| # | Category | What is measured |
|---|----------|-----------------|
| 1 | **Acceptance** | Unconditional positive regard communicated by the counselor |
| 2 | **Empathy** | Accuracy and consistency of reflective listening |
| 3 | **Direction** | Degree to which the counselor maintains focus on the target behavior |
| 4 | **Collaboration** | Equal-partnership stance between counselor and client |
| 5 | **Autonomy Support** | Active fostering of client's sense of choice |
| 6 | **Evocation** | Extent to which the counselor draws out the client's own motivation |
| 7 | **Partnership** | Recognition that expertise and wisdom reside within the client |
| 8 | **Cultivating Change Talk** | Skill at eliciting and strengthening change-oriented language |
| 9 | **Softening Sustain Talk** | Skill at de-emphasizing resistance or sustain talk |

Each criterion is rated from 1 (lowest) to 5 (highest).
An **Overall Average** across all nine criteria is also computed per session.

---

## 2. Data Structure

Each automated test run produces an `MI_aggregated_<timestamp>/` folder under
`saved_tests/` containing three score CSVs:

| File | Contents |
|------|----------|
| `good_student_all.csv` | One row per patient — scores from the high-competence (good) student |
| `bad_student_1st_all.csv` | One row per patient — scores from the low-competence student's **first** attempt |
| `bad_student_feedback_all.csv` | One row per patient — scores from the low-competence student's **second** attempt (after receiving evaluator feedback) |

Each CSV has columns: `Student`, `Patient`, and one column per MI category, plus `Overall Average`.

The scripts always read from the **most recently timestamped** `MI_aggregated_*` folder.

---

## 3. Hypothesis Tests

### Hypothesis 1 — Expert vs. Novice (H1)

**Research question:** Do good (expert-simulated) students score higher than bad
(novice-simulated) students across all MI categories?

**Null hypothesis H₀:** The distributions of scores for the good student and the bad
student are identical for each MI category.

**Alternative hypothesis H₁:** The good student scores are stochastically greater
than the bad student scores.

**Design:** Each patient profile is used once for both student types within the same
test run, creating a natural one-to-one pairing. Scores are paired by the `Patient`
column via an inner join.

**Test:** Two-sided Wilcoxon Signed-Rank test (`scipy.stats.wilcoxon`).

$$
H_0: P(X_{\text{good}} > X_{\text{bad}}) = P(X_{\text{bad}} > X_{\text{good}})
$$

A two-sided test is used because the direction of the expected difference (good > bad)
is verified empirically from the mean difference, rather than assumed.

---

### Hypothesis 2 — Learning Effect (H2)

**Research question:** Does the low-competence student improve after receiving
structured evaluator feedback?

**Null hypothesis H₀:** Scores on the second attempt are not greater than scores on
the first attempt.

**Alternative hypothesis H₁:** Scores on the second attempt are stochastically
greater than scores on the first attempt (i.e., first < second).

**Design:** The same patient is used for both the first and second attempt by the same
bad student, creating a one-to-one pairing. Scores are paired by the `Patient` column.

**Test:** One-sided Wilcoxon Signed-Rank test (`alternative="less"`), testing whether
the first-attempt scores are systematically lower than the second-attempt scores.

$$
H_0: P(X_{\text{first}} < X_{\text{second}}) \leq 0.5
$$

---

## 4. Wilcoxon Signed-Rank Test

The Wilcoxon Signed-Rank test is applied **independently for each MI category**
(including Overall Average), yielding 10 tests per hypothesis.

### Why Wilcoxon instead of a paired t-test?

The evaluation rubric produces ordinal Likert scores (1–5), which may not follow a
normal distribution. The Wilcoxon Signed-Rank test is a non-parametric alternative
to the paired t-test that is appropriate for ordinal data.

### Preprocessing within each category

For each category, let $x_i$ be the first group's score and $y_i$ be the second
group's score for patient $i$.

1. Any pair where either $x_i$ or $y_i$ is `NaN` (missing) is excluded.
2. For H1 only: any pair where $x_i = y_i$ (zero difference) is also excluded,
   because Wilcoxon cannot rank a zero difference.
3. The test proceeds only if at least 2 valid pairs remain.

### Test statistic

The Wilcoxon statistic $W$ is the sum of the signed ranks of the non-zero
differences:

$$
W = \sum_{i: d_i \neq 0} \text{sgn}(d_i) \cdot R_i
$$

where $d_i = x_i - y_i$ and $R_i$ is the rank of $|d_i|$ among all non-zero
absolute differences.

`scipy.stats.wilcoxon` returns the smaller of the two one-sided statistics.

---

## 5. Effect Size: Rank-Biserial Correlation

For each Wilcoxon test, the effect size is computed from the test statistic $W$
and the number of valid pairs $n$:

$$
r = 1 - \frac{2W}{n(n+1)}
$$

This is the rank-biserial correlation coefficient, the natural effect size for the
Wilcoxon Signed-Rank test. It is stored as `Effect_Size_r`.

### Interpretation

| $r$ | Interpretation |
|-----|---------------|
| $\approx 1$ | All differences are in the same (expected) direction — maximum effect |
| $\approx 0$ | Differences are balanced between directions — no effect |
| $\approx -1$ | All differences are in the opposite direction |

For H1, $r = 1.0$ across all categories indicates that the good student scored
strictly higher than the bad student on every single patient without exception.

For H2, $r$ values in the range 0.75–0.99 indicate large positive learning effects
across all categories.

---

## 6. Multiple Comparisons Correction

Because each script tests 10 categories (9 MI criteria + Overall Average) for a
single comparison, Benjamini–Hochberg False Discovery Rate (BH-FDR) correction is
applied across the 10 p-values within each script.

### Procedure

Let $p_1, p_2, \dots, p_{10}$ be the raw Wilcoxon p-values sorted in ascending order.
For each $p_{(k)}$:

$$
p_{\text{FDR},(k)} = \min\left(1,\ \frac{m \cdot p_{(k)}}{k}\right)
$$

where $m = 10$ is the total number of tests.

This is implemented using:

```python
from statsmodels.stats.multitest import multipletests
reject, p_fdr, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
```

### Why BH-FDR and not Bonferroni?

Bonferroni controls the family-wise error rate (FWER), which is appropriate when
any false positive is unacceptable. Here, the 10 MI categories are positively
correlated (a student who scores well on Empathy also tends to score well on
Acceptance). BH-FDR is less conservative under positive dependence, provides higher
statistical power, and is the more appropriate choice when the tests share a
common underlying construct.