"""Test suite for the AI-assisted operations workflow optimizer."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_data import generate_tickets, tickets_to_dicts, CATEGORIES
from classifier import train_and_evaluate, held_out_template_ids, TicketClassifier
from workflow_analysis import (
    avg_resolution_by_team, avg_resolution_by_category, backlog_by_team,
    ticket_volume_by_team, worst_bottleneck,
)
from optimizer import simulate_reallocation, overall_avg_resolution


@pytest.fixture
def small_data():
    tickets = generate_tickets(n=500, seed=1)
    return tickets_to_dicts(tickets)


@pytest.fixture
def full_data():
    tickets = generate_tickets(n=2000, seed=42)
    return tickets_to_dicts(tickets)


# --- Data generation ---------------------------------------------------------

def test_generates_requested_number_of_tickets(small_data):
    assert len(small_data) == 500


def test_every_ticket_has_a_valid_category(small_data):
    valid_categories = set(CATEGORIES.keys())
    assert all(t["category"] in valid_categories for t in small_data)


def test_open_tickets_have_no_resolution_time(small_data):
    for t in small_data:
        if t["status"] == "Open":
            assert t["resolution_hours"] is None
        else:
            assert t["resolution_hours"] is not None


def test_resolution_hours_never_negative(small_data):
    for t in small_data:
        if t["resolution_hours"] is not None:
            assert t["resolution_hours"] > 0


def test_generation_is_deterministic_given_same_seed():
    a = tickets_to_dicts(generate_tickets(n=200, seed=7))
    b = tickets_to_dicts(generate_tickets(n=200, seed=7))
    assert a == b


def test_different_seeds_produce_different_data():
    a = tickets_to_dicts(generate_tickets(n=200, seed=1))
    b = tickets_to_dicts(generate_tickets(n=200, seed=2))
    assert a != b


# --- Classifier: template-level split correctness ---------------------------

def test_held_out_templates_cover_every_category():
    held = held_out_template_ids()
    assert set(held.keys()) == set(CATEGORIES.keys())
    for cat, ids in held.items():
        assert len(ids) >= 1
        assert all(0 <= i < len(CATEGORIES[cat]) for i in ids)


def test_train_test_split_has_no_template_overlap(full_data):
    result = train_and_evaluate(full_data)
    held = result["held_out_templates"]
    train_rows = [t for t in full_data if t["template_id"] not in held[t["category"]]]
    test_rows = [t for t in full_data if t["template_id"] in held[t["category"]]]
    train_template_keys = {(t["category"], t["template_id"]) for t in train_rows}
    test_template_keys = {(t["category"], t["template_id"]) for t in test_rows}
    assert train_template_keys.isdisjoint(test_template_keys)


def test_classifier_beats_random_baseline(full_data):
    """5 roughly-balanced categories -> random guessing is ~20%. The
    classifier must clear that bar by a wide margin on genuinely unseen
    phrasings, or it isn't demonstrating real generalization."""
    result = train_and_evaluate(full_data)
    assert result["accuracy"] > 0.5


def test_classifier_accuracy_is_not_suspiciously_perfect(full_data):
    """A row-level split over a small finite template pool can trivially
    hit ~100% by memorization — this guards against ever regressing back
    to that meaningless setup."""
    result = train_and_evaluate(full_data)
    assert result["accuracy"] < 0.99


def test_per_class_metrics_have_every_category(full_data):
    result = train_and_evaluate(full_data)
    assert set(result["per_class"].keys()) == set(CATEGORIES.keys())


def test_classifier_predict_returns_valid_labels(full_data):
    result = train_and_evaluate(full_data)
    clf = result["classifier"]
    preds = clf.predict(["My screen is flickering and the keyboard is unresponsive"])
    assert preds[0] in CATEGORIES.keys()


# --- Workflow analysis --------------------------------------------------------

def test_avg_resolution_by_team_matches_manual_calculation(full_data):
    computed = avg_resolution_by_team(full_data)
    # Manually recompute for one team.
    a_team = next(iter(computed))
    manual_vals = [t["resolution_hours"] for t in full_data if t["team"] == a_team and t["status"] == "Resolved"]
    manual_avg = round(sum(manual_vals) / len(manual_vals), 2)
    assert computed[a_team] == manual_avg


def test_avg_resolution_by_category_matches_manual_calculation(full_data):
    computed = avg_resolution_by_category(full_data)
    a_cat = next(iter(computed))
    manual_vals = [t["resolution_hours"] for t in full_data if t["category"] == a_cat and t["status"] == "Resolved"]
    manual_avg = round(sum(manual_vals) / len(manual_vals), 2)
    assert computed[a_cat] == manual_avg


def test_backlog_only_counts_open_tickets(full_data):
    backlog = backlog_by_team(full_data)
    total_backlog = sum(backlog.values())
    manual_open_count = sum(1 for t in full_data if t["status"] == "Open")
    assert total_backlog == manual_open_count


def test_ticket_volume_matches_total_ticket_count(full_data):
    volumes = ticket_volume_by_team(full_data)
    assert sum(volumes.values()) == len(full_data)


def test_worst_bottleneck_is_actually_the_max(full_data):
    result = worst_bottleneck(full_data)
    by_combo = {}
    from collections import defaultdict
    totals = defaultdict(list)
    for t in full_data:
        if t["status"] == "Resolved":
            totals[(t["team"], t["category"])].append(t["resolution_hours"])
    candidates = {k: sum(v) / len(v) for k, v in totals.items() if len(v) >= 5}
    manual_worst_key = max(candidates, key=candidates.get)
    assert result["team"] == manual_worst_key[0]
    assert result["category"] == manual_worst_key[1]


# --- Optimizer / reallocation simulation -------------------------------------

def test_reallocation_moves_tickets_from_overloaded_to_underloaded(full_data):
    sim = simulate_reallocation(full_data)
    volumes = ticket_volume_by_team(full_data)
    assert sim["overloaded_team"] == max(volumes, key=volumes.get)
    assert sim["underloaded_team"] == min(volumes, key=volumes.get)


def test_reallocation_improves_average_resolution_time(full_data):
    """The whole point of the simulated reallocation is that it should
    genuinely improve the average — this test would fail if the logic
    were reversed or broken, not just check that it runs."""
    sim = simulate_reallocation(full_data)
    assert sim["after_avg_resolution_hours"] < sim["before_avg_resolution_hours"]
    assert sim["improvement_pct"] > 0


def test_reallocation_respects_move_fraction(full_data):
    from workflow_analysis import _resolved
    resolved = _resolved(full_data)
    sim = simulate_reallocation(full_data, move_fraction=0.1)
    volumes = ticket_volume_by_team(full_data)
    overloaded = max(volumes, key=volumes.get)
    resolved_overloaded_count = sum(1 for t in resolved if t["team"] == overloaded)
    expected = round(resolved_overloaded_count * 0.1)
    assert sim["n_tickets_moved"] == expected


def test_reallocation_raises_with_only_one_team():
    tickets = generate_tickets(n=50, seed=1)
    data = tickets_to_dicts(tickets)
    for t in data:
        t["team"] = "OnlyTeam"
    with pytest.raises(ValueError):
        simulate_reallocation(data)


def test_overall_avg_resolution_matches_manual_calculation(full_data):
    computed = overall_avg_resolution(full_data)
    resolved_vals = [t["resolution_hours"] for t in full_data if t["status"] == "Resolved"]
    manual = round(sum(resolved_vals) / len(resolved_vals), 2)
    assert computed == manual
