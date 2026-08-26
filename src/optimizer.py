"""A concrete, data-backed reallocation recommendation: given the current
team workload imbalance, simulates moving a slice of the most-overloaded
team's tickets to underloaded teams, and measures the resulting change in
average resolution time. This is the "optimization" piece — a specific,
testable recommendation with a measured effect, not a vague suggestion.
"""
from workflow_analysis import avg_resolution_by_team, ticket_volume_by_team, _resolved


def overall_avg_resolution(tickets_data):
    resolved = _resolved(tickets_data)
    if not resolved:
        return None
    return round(sum(t["resolution_hours"] for t in resolved) / len(resolved), 2)


def simulate_reallocation(tickets_data, move_fraction=0.25, random_state=42):
    """Simulates moving move_fraction of the most-overloaded team's
    RESOLVED tickets to the least-loaded team, and recomputes what the
    average resolution time for those specific tickets WOULD have been
    had they been handled at the receiving team's typical pace (modeled
    via that team's own average resolution time for the same category
    where available, falling back to the team's overall average).

    This is a simulation over historical data, not a live system — it
    answers "if we'd routed a slice of this team's load elsewhere, would
    the average have improved, and by how much?" with an actual number.
    """
    import random
    rng = random.Random(random_state)

    volumes = ticket_volume_by_team(tickets_data)
    team_avgs = avg_resolution_by_team(tickets_data)
    if len(volumes) < 2:
        raise ValueError("Need at least 2 teams to simulate reallocation")

    overloaded_team = max(volumes, key=volumes.get)
    underloaded_team = min(volumes, key=volumes.get)
    if overloaded_team == underloaded_team:
        raise ValueError("Cannot reallocate: overloaded and underloaded team are the same")

    resolved = _resolved(tickets_data)
    overloaded_tickets = [t for t in resolved if t["team"] == overloaded_team]

    n_to_move = round(len(overloaded_tickets) * move_fraction)
    moved = rng.sample(overloaded_tickets, min(n_to_move, len(overloaded_tickets)))
    moved_ids = {t["ticket_id"] for t in moved}

    # Category-specific average for the receiving team, where we have
    # enough data; otherwise fall back to that team's overall average.
    from collections import defaultdict
    receiving_by_category = defaultdict(list)
    for t in resolved:
        if t["team"] == underloaded_team:
            receiving_by_category[t["category"]].append(t["resolution_hours"])
    receiving_category_avg = {
        cat: sum(vals) / len(vals) for cat, vals in receiving_by_category.items() if len(vals) >= 3
    }
    receiving_overall_avg = team_avgs[underloaded_team]

    before_total = sum(t["resolution_hours"] for t in resolved)
    after_total = 0.0
    for t in resolved:
        if t["ticket_id"] in moved_ids:
            simulated_time = receiving_category_avg.get(t["category"], receiving_overall_avg)
            after_total += simulated_time
        else:
            after_total += t["resolution_hours"]

    before_avg = round(before_total / len(resolved), 2)
    after_avg = round(after_total / len(resolved), 2)
    improvement_pct = round((before_avg - after_avg) / before_avg * 100, 2) if before_avg else 0.0

    return {
        "overloaded_team": overloaded_team,
        "underloaded_team": underloaded_team,
        "n_tickets_moved": len(moved),
        "before_avg_resolution_hours": before_avg,
        "after_avg_resolution_hours": after_avg,
        "improvement_pct": improvement_pct,
    }
