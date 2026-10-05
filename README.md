# AI-Assisted Operations Workflow Optimizer

A tested project combining a trained ML text classifier with operations-workflow bottleneck analysis and a data-backed reallocation recommendation. It puts an AI component (automatic ticket triage) and operations-workflow optimization together in one pipeline.

## What it does

- **`src/generate_data.py`**: generates ~2,000 synthetic operations tickets (IT/access-request/onboarding-style: category, free-text description, assigned team, priority, created/resolved timestamps, status), with realistic backlog and resolution-time variation built in (some teams and categories are deliberately slower, to give the bottleneck analysis something to find).
- **`src/classifier.py`**: trains a TF-IDF + `LogisticRegression` classifier to predict a ticket's category from its free-text description alone (the auto-triage piece), evaluated on a held-out test split with accuracy, per-class precision/recall, and a confusion matrix.
- **`src/workflow_analysis.py`**: computes bottleneck metrics from the ticket data: average resolution time by category and by team, backlog depth (open tickets) by team, and the single worst-performing team/category combination.
- **`src/optimizer.py`**: a concrete, data-backed recommendation. Given the current team workload imbalance, it simulates reallocating a slice of tickets from the most-overloaded team to underloaded ones and measures the resulting change in average resolution time.
- **`src/pipeline.py`**: runs the full flow end to end and prints a combined report (classifier performance, bottleneck findings, and the reallocation recommendation with its measured effect).

## Data

The dataset is synthetic, generated to be realistic (ticket categories, descriptions, team assignments, timestamps, resolution times), and the pipeline is built so real ticket exports can replace it.

The AI component is a trained, evaluated scikit-learn text classifier (TF-IDF + logistic regression) rather than an LLM. It trains on the synthetic ticket descriptions with a held-out test split and reports accuracy and F1 on data it never saw during training.

## Results

Evaluation uses a **template-level split**. The ticket descriptions come from a finite pool of phrasing templates, so a plain row-level train/test split would let the model memorize templates and score a meaningless ~100% accuracy. Instead, a third of each category's description templates are held out entirely from training, so every test-set phrasing is one the model never saw. On this held-out evaluation the classifier scores **78.6% accuracy** (random-guess baseline for 5 categories is ~20%), with per-category variation: it generalizes very well on some categories (`onboarding`: 100% F1) and is weaker on others (`software_bug`: 42% F1, where held-out phrasings lexically overlap with other categories like login/access issues).

## Tests

22 automated tests (`tests/test_ops_workflow.py`) covering:

- Data generation shape and determinism.
- The template-level split's correctness (zero overlap between train and test templates).
- Classifier accuracy against both a random-baseline floor and a suspiciously-perfect-score ceiling (guarding against regressing to the memorization setup).
- Workflow-analysis metrics against manually recomputed values.
- The reallocation simulation's correctness: moves come from the actually-overloaded team, the move count matches the requested fraction, and the simulated average resolution time improves rather than being asserted to.

## Project structure

```
src/
  generate_data.py
  classifier.py
  workflow_analysis.py
  optimizer.py
  pipeline.py
tests/
  test_ops_workflow.py
```

## Running it

```bash
pip install scikit-learn pandas numpy
python3 src/pipeline.py          # generates data, trains classifier, runs analysis, prints full report
python3 -m pytest tests/ -v      # runs all 22 tests
```

## Notes

The classifier is classical ML, a good fit for a short-text, five-category triage task. The classifier step is isolated in `src/classifier.py` so a larger language model can be swapped in behind the same interface.

## Possible extensions

- Replace or augment the TF-IDF classifier with an LLM-based triage step and compare on the same template-level split.
- Add priority prediction and an estimated resolution-time model.
- Extend the reallocation simulation to account for team skills and capacity.
