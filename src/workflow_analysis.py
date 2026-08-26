"""Bottleneck analysis over the ticket data — the "operations workflow
optimization" piece. Computes resolution-time and backlog metrics by team
and category to find where the real slowdowns are.
"""
from collections import defaultdict


def _resolved(tickets_data):
    return [t for t in tickets_data if t["status"] == "Resolved"]


def avg_resolution_by_team(tickets_data):
    totals = defaultdict(list)
    for t in _resolved(tickets_data):
        totals[t["team"]].append(t["resolution_hours"])
    return {team: round(sum(vals) / len(vals), 2) for team, vals in totals.items()}


def avg_resolution_by_category(tickets_data):
    totals = defaultdict(list)
    for t in _resolved(tickets_data):
        totals[t["category"]].append(t["resolution_hours"])
    return {cat: round(sum(vals) / len(vals), 2) for cat, vals in totals.items()}


def backlog_by_team(tickets_data):
    """Count of currently-Open tickets per team — the queue depth a
    real operations dashboard would surface."""
    counts = defaultdict(int)
    for t in tickets_data:
        if t["status"] == "Open":
            counts[t["team"]] += 1
    return dict(counts)


def ticket_volume_by_team(tickets_data):
    counts = defaultdict(int)
    for t in tickets_data:
        counts[t["team"]] += 1
    return dict(counts)


def worst_bottleneck(tickets_data):
    """Returns the single (team, category) combination with the highest
    average resolution time among combinations with at least min_count
    resolved tickets, so a single slow outlier ticket doesn't dominate."""
    totals = defaultdict(list)
    for t in _resolved(tickets_data):
        totals[(t["team"], t["category"])].append(t["resolution_hours"])

    min_count = 5
    worst = None
    worst_avg = -1
    for key, vals in totals.items():
        if len(vals) < min_count:
            continue
        avg = sum(vals) / len(vals)
        if avg > worst_avg:
            worst_avg = avg
            worst = key

    if worst is None:
        return None
    return {"team": worst[0], "category": worst[1], "avg_resolution_hours": round(worst_avg, 2), "n_tickets": len(totals[worst])}
