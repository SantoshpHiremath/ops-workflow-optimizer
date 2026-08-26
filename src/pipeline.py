"""End-to-end pipeline: generate synthetic tickets, train/evaluate the
classifier, run the bottleneck analysis, run the reallocation simulation,
and print a combined operational report.
"""
from generate_data import generate_tickets, tickets_to_dicts
from classifier import train_and_evaluate
from workflow_analysis import (
    avg_resolution_by_team, avg_resolution_by_category, backlog_by_team,
    ticket_volume_by_team, worst_bottleneck,
)
from optimizer import simulate_reallocation, overall_avg_resolution


def run():
    tickets = generate_tickets()
    data = tickets_to_dicts(tickets)

    print(f"Loaded {len(data)} synthetic operations tickets.\n")

    print("=== Ticket Auto-Triage Classifier (TF-IDF char n-grams + LogisticRegression) ===")
    clf_result = train_and_evaluate(data)
    print(f"Trained on {clf_result['n_train']} tickets, evaluated on {clf_result['n_test']} tickets")
    print(f"using phrasings the model NEVER saw during training (template-level split).")
    print(f"Overall accuracy: {clf_result['accuracy'] * 100:.1f}%")
    for label, stats in clf_result["per_class"].items():
        print(f"  {label}: precision={stats['precision']}, recall={stats['recall']}, f1={stats['f1']} (n={stats['support']})")

    print("\n=== Workflow Bottleneck Analysis ===")
    print("Avg resolution time by team (hours):", avg_resolution_by_team(data))
    print("Avg resolution time by category (hours):", avg_resolution_by_category(data))
    print("Current backlog (open tickets) by team:", backlog_by_team(data))
    print("Ticket volume by team:", ticket_volume_by_team(data))
    bottleneck = worst_bottleneck(data)
    print(f"Worst bottleneck: {bottleneck['team']} / {bottleneck['category']} — "
          f"{bottleneck['avg_resolution_hours']}h avg across {bottleneck['n_tickets']} tickets")

    print("\n=== Reallocation Recommendation (simulated) ===")
    sim = simulate_reallocation(data)
    print(f"Overloaded team: {sim['overloaded_team']} -> Underloaded team: {sim['underloaded_team']}")
    print(f"Simulated moving {sim['n_tickets_moved']} tickets")
    print(f"Overall avg resolution time: {sim['before_avg_resolution_hours']}h -> {sim['after_avg_resolution_hours']}h "
          f"({sim['improvement_pct']}% improvement)")


if __name__ == "__main__":
    run()
