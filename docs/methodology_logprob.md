# Virtual Patient Consistency: Log-Probability Methodology

This document describes the mathematical framework used to quantify the
behavioural consistency of the virtual patient (GPT-4.1-mini) across simulated
therapy sessions in the LearningBuddy framework.

---

## 1. Motivation

The virtual patient is a language model prompted to roleplay a specific clinical
profile. A key validity question is: **does the patient respond consistently to
different students, or does it drift arbitrarily?**

Rather than a separate human-rating pass, we exploit the token-level log-
probabilities that the model itself assigns during generation. These capture how
"certain" the model was about each token it produced — a proxy for how
constrained the patient's behaviour is by its persona prompt.

---

## 2. Token Log-Probability

For a generated sequence $y = (y_1, y_2, \ldots, y_T)$ the model assigns a probability to each token conditioned on all preceding context:

$$p(y_t \mid y_{<t}, x)$$

where $x$ is the full prompt (system prompt + conversation history). The **token log-probability** is:

$$\ell_t = \log p(y_t \mid y_{<t}, x)$$

Because $0 < p \leq 1$, we have $\ell_t \leq 0$ always.

---

## 3. Turn-Level Average Log-Probability

For a single patient utterance of $T$ tokens the **turn average log-probability**
is the arithmetic mean over tokens:

$$\bar{\ell}_{\text{turn}} = \frac{1}{T} \sum_{t=1}^{T} \ell_t$$

This equals the per-token log-likelihood of the utterance under the model.

### Clipping

OpenAI returns $\ell_t = -9999$ as a sentinel for tokens that were generated
despite having near-zero probability (typically forced special tokens such as
role delimiters). These sentinels are not meaningful probability estimates and
would dominate the average. We therefore clip each token logprob at $-10$
before averaging:

$$\tilde{\ell}_t = \max(\ell_t,\ -10)$$

The threshold $-10$ corresponds to a token probability of $e^{-10} \approx
0.005\%$, which is effectively zero. Clipping does not affect any token that
carries real information.

$$\bar{\ell}_{\text{turn}} = \frac{1}{T} \sum_{t=1}^{T} \tilde{\ell}_t$$

---

## 4. Session-Level Average Log-Probability

Each session consists of $N = 10$ patient turns. The **session average log-
probability** is the mean of the turn averages:

$$\bar{\ell}_{\text{session}} = \frac{1}{N} \sum_{n=1}^{N} \bar{\ell}_{\text{turn},\, n}$$

This is stored as `patient_avg_logprob` in `MI_scores.csv` and as
`session_avg_logprob` in `patient_logprobs.json`.

---

## 5. Geometric Mean Token Probability

For reporting purposes we also convert to a percentage that is easier to
interpret:

$$p_{\text{avg}} = e^{\bar{\ell}_{\text{session}}} \times 100\%$$

This is the **geometric mean of the per-token probabilities** across the session.
It answers: *"On average, how likely was each token the patient generated?"*

| $p_{\text{avg}}$ | $\bar{\ell}_{\text{session}}$ | Interpretation |
|---|---|---|
| ~60 % | ~−0.5 | High confidence — patient behaviour is tightly constrained by the persona |
| ~37 % | −1.0 | Moderate confidence |
| ~14 % | −2.0 | Low confidence — patient behaviour is more free-form |
| <1 % | < −5 | Indicates a sentinel-contaminated turn slipped through |

Stored as `patient_avg_prob_pct` in `MI_scores.csv` and `session_avg_prob_pct` in
`patient_logprobs.json`.

---

## 6. Relationship to Perplexity

The session average log-probability relates directly to **perplexity**, the
standard NLP metric for language model uncertainty:

$$\text{PPL}_{\text{session}} = e^{-\bar{\ell}_{\text{session}}}$$

Lower perplexity means the model found the generated text more predictable given
the prompt. Because we report $\bar{\ell}_{\text{session}}$ (negative), higher
(less negative) values correspond to lower perplexity and higher consistency.

---

## 7. Saved Artefacts

For every session `run_test()` writes a `patient_logprobs.json` alongside the
other session files:

```json
{
  "turn_logprobs": [-0.65, -0.71, ...],   // N floats, one per turn
  "session_avg_logprob": -0.6887,          // mean of turn_logprobs
  "session_avg_prob_pct": 50.2,            // exp(session_avg_logprob) × 100
  "n_turns": 10
}
```

`MI_scores.csv` appends two columns to each patient row:

| Column | Type | Description |
|---|---|---|
| `patient_avg_logprob` | float | $\bar{\ell}_{\text{session}}$, rounded to 4 d.p. |
| `patient_avg_prob_pct` | float | $p_{\text{avg}}$, rounded to 2 d.p. |

---

## 8. Relationship to Model Confidence and Hallucination

### 8.1 Log-Probability as a Confidence Signal

The token log-probability $\ell_t$ is the model's own estimate of how likely it
was to produce token $y_t$ given everything that came before it. In this sense it
is a **first-person confidence score**: a high (less negative) value means the
model was "sure" the next token should be $y_t$; a low (very negative) value
means the model generated something it considered unlikely.

Crucially, this is not just syntactic confidence — it reflects the model's
internal calibration between the prompt constraints (the patient persona) and the
generated text. When the persona is specific and consistent, the model's
generation is more deterministic and logprobs are higher. When the prompt is
ambiguous, generation is more stochastic and logprobs are lower.

### 8.2 Hallucination and Low Log-Probability

Hallucination in roleplay LLMs refers to the patient generating content that
contradicts its own case profile — fabricating facts, switching personality, or
breaking character. Research has shown that hallucinated tokens tend to receive
**lower log-probabilities** than factually grounded ones, because the model is
producing output that sits in a low-density region of the distribution conditioned
on the prompt.

This gives logprobs diagnostic power:

| Observation | Likely cause |
|---|---|
| Single turn with very low $\bar\ell_{\text{turn}}$ | One hallucinated or out-of-character utterance |
| Persistently low $\bar\ell_{\text{session}}$ across all turns | Patient profile is under-specified or internally inconsistent |
| Low logprobs only with bad students | Patient is forced into unexpected territory by poor counselling — desirable sensitivity |
| Low logprobs even with good students | Patient profile may need revision |

### 8.3 Limitations

Log-probability is a **necessary but not sufficient** condition for detecting
hallucination:

1. **Fluent confabulation**: a model can produce a confident ($\ell_t$ close to
   0) but factually wrong token if wrong content is the most probable completion
   of the prompt. This is the "confident hallucination" failure mode and is not
   caught by logprobs alone.

2. **Calibration varies by model**: GPT-4.1-mini's logprobs are well-calibrated
   but not perfect. The absolute values should be interpreted relative to a
   baseline (e.g., the same patient profile with a neutral/scripted student)
   rather than against an absolute threshold.

3. **Token-level vs. semantic-level**: a single rare token (e.g., a proper noun
   from the case profile) can lower $\bar\ell_{\text{turn}}$ even when the
   overall utterance is semantically correct. This is why we clip sentinels and
   report session-level averages rather than individual tokens.

Despite these limitations, session-average logprob remains a lightweight,
zero-annotation signal for **flagging sessions that warrant closer human review**,
particularly in large-scale automated experiments where reading every transcript
is infeasible.
