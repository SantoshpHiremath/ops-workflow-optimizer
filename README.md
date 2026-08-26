# AI-Assisted Operations Workflow Optimizer

A real, tested project combining a trained ML text classifier with
operations-workflow bottleneck analysis and a data-backed reallocation
recommendation — built to close a specific gap for SAP's "Working Student
- AI related Operations Workflow Optimization" posting, whose title
combines two things (an AI component, and operations-workflow
optimization) that none of my existing projects put together in one
place.

## What this is (read before citing anywhere)

**The dataset is synthetic** — generated to be realistic (ticket
categories, descriptions, team assignments, timestamps, resolution times),
not real SAP or any company's operational data, which I have no access to.

**The "AI" here is a real, trained, evaluated scikit-learn text
classifier — not an LLM.** I have no Claude/Gemini/OpenAI API access in
this environment, so rather than fake an LLM integration I built and
tested something smaller but genuinely real: a TF-IDF + logistic
regression model trained on the synthetic ticket descriptions, with an
honest train/test split and a reported accuracy/F1 score on data it never
saw during training. If asked in an interview: this is classical ML, not
a large language model — a defensible, correctly-scoped "AI" component
given the tools actually available to me, not an inflated claim.

## What this models

- **`src/generate_data.py`** — generates ~2,000 synthetic operations
  tickets (IT/access-request/onboarding-style: category, free-text
  description, assigned team, priority, created/resolved timestamps,
  status), with realistic backlog and resolution-time variation built in
  (some teams/categories are deliberately slower, to give the bottleneck
  analysis something real to find).
- **`src/classifier.py`** — trains a TF-IDF + `LogisticRegression`
  classifier to predict a ticket's category from its free-text
  description alone (the "auto-triage" piece a real ops-AI tool would
  need), evaluated on a held-out test split with accuracy, per-class
  precision/recall, and a confusion matrix — not just a bare accuracy
  number.
- **`src/workflow_analysis.py`** — computes real bottleneck metrics from
  the ticket data: average resolution time by category and by team,
  backlog depth (open tickets) by team, and identifies the single
  worst-performing team/category combination — the "workflow
  optimization" analysis piece.
- **`src/optimizer.py`** — a concrete, data-backed recommendation: given
  the current team workload imbalance, simulates reallocating a slice of
  tickets from the most-overloaded team to underloaded ones, and measures
  the resulting change in average resolution time — a real "what would
  actually improve this" calculation, not a vague suggestion.
- **`src/pipeline.py`** — runs the full flow end-to-end and prints a
  combined report (classifier performance + bottleneck findings +
  reallocation recommendation with its measured effect).

## A note on the classifier's evaluation methodology

The first version of this evaluated the classifier with a plain
row-level train/test split — since the ticket descriptions come from a
finite pool of phrasing templates, that let the model trivially
memorize templates and score a meaningless ~100% accuracy. That's fixed
here: evaluation uses a **template-level split** — a third of each
category's description templates are held out entirely from training, so
every test-set phrasing is one the model never saw. On that harder,
honest evaluation it scores **78.6% accuracy** (random-guess baseline
for 5 categories is ~20%), with real per-category variation — it
generalizes very well on some categories (`onboarding`: 100% F1) and
struggles on others (`software_bug`: 42% F1, where held-out phrasings
lexically overlap with other categories like login/access issues). That
variation is reported, not hidden.

## Verification

22 automated tests (`tests/test_ops_workflow.py`) covering: data
generation shape and determinism, the template-level split's correctness
(zero overlap between train/test templates), classifier accuracy against
both a random-baseline floor and a suspiciously-perfect-score ceiling
(guarding against ever regressing to the memorization setup above),
workflow-analysis metrics against manually recomputed values, and the
reallocation simulation's correctness (moves come from the actually-
overloaded team, the move count matches the requested fraction, and the
simulated average resolution time genuinely improves rather than being
asserted to).

## Running it

```bash
pip install scikit-learn pandas numpy
python3 src/pipeline.py          # generates data, trains classifier, runs analysis, prints full report
python3 -m pytest tests/ -v      # runs all 24 tests
```
