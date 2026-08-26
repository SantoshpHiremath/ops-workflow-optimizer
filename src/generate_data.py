"""Generates a SYNTHETIC operations-ticket dataset — not real SAP or any
company's data. Modeled on IT/access-request/onboarding-style internal
operations tickets, with realistic team/category resolution-time
imbalance deliberately built in so the bottleneck analysis has real
signal to find, not a uniformly-fast synthetic dataset.

Each ticket also records which description TEMPLATE it came from
(template_id) — this lets the classifier be evaluated with a genuine
template-level train/test split (some phrasings held out entirely from
training) rather than a naive row-level split, which would let the model
trivially memorize a small, finite set of templates and score an
artificially perfect (and meaningless) accuracy. See classifier.py.
"""
import random
from dataclasses import dataclass, asdict

CATEGORIES = {
    "access_request": [
        "Need access to the shared drive for the {team} project",
        "Requesting VPN access for remote work starting next week",
        "Please grant admin rights on my laptop for software install",
        "Access to the internal wiki is showing permission denied",
        "New hire needs badge access to building {n}",
        "Can someone reset my password, I'm locked out of my account",
        "Requesting access to the customer database for reporting",
        "My SSO login keeps failing after the recent update",
        "Please add me to the {team} shared mailbox",
        "I need elevated permissions to deploy to the staging environment",
        "Guest wifi access needed for a visiting contractor",
        "Two-factor authentication device was lost, need to re-enroll",
    ],
    "hardware_issue": [
        "My laptop won't turn on after the update",
        "Docking station stopped charging my device",
        "Monitor showing flickering screen since this morning",
        "Keyboard keys are unresponsive on the left side",
        "Need a replacement charger, mine stopped working",
        "The conference room projector isn't detecting my laptop",
        "Headset microphone is not being picked up in calls",
        "External hard drive is not being recognized by the system",
        "Printer on the {n}th floor keeps jamming paper",
        "Trackpad has become unresponsive after the last restart",
        "Webcam shows a black screen during video calls",
        "Laptop fan is unusually loud and the device overheats",
    ],
    "software_bug": [
        "The reporting tool crashes when I export to Excel",
        "Login page throws a 500 error intermittently",
        "Calendar sync is duplicating meeting invites",
        "Search feature returns no results for valid queries",
        "Application freezes when opening large files",
        "Notifications stopped appearing after the latest release",
        "Dashboard widgets show stale data even after refreshing",
        "File upload fails silently for files over 10 MB",
        "Dark mode toggle resets every time I log back in",
        "Timezone displayed is wrong for all scheduled meetings",
        "The mobile app logs me out randomly during use",
        "Autosave isn't triggering, I lost unsaved changes twice",
    ],
    "onboarding": [
        "New employee starting Monday needs equipment setup",
        "Please add new hire to the team distribution list",
        "Onboarding checklist item: assign a buddy/mentor",
        "New starter needs accounts provisioned before day one",
        "Requesting orientation session scheduling for new team member",
        "Can we schedule a welcome call for the incoming intern",
        "New hire paperwork is missing a signature, please follow up",
        "Requesting a desk assignment for the new team member",
        "Please share the onboarding handbook with the new starter",
        "New employee needs to be added to the relevant Slack channels",
        "Setting up a first-week schedule for the new graduate hire",
        "New team member is missing from the org chart, please update",
    ],
    "hr_query": [
        "Question about remaining vacation days for this year",
        "Need clarification on the expense reimbursement process",
        "Requesting an update to my emergency contact information",
        "Asking about parental leave policy details",
        "Payslip from last month seems incorrect, please check",
        "How do I update my bank details for payroll",
        "Question about the tuition reimbursement program eligibility",
        "Requesting a copy of my employment contract",
        "Asking whether sick days carry over into next year",
        "Need guidance on the process for requesting a sabbatical",
        "Question about health insurance enrollment deadlines",
        "Requesting an official letter confirming my employment",
    ],
}

TEAMS = ["IT-Support", "Facilities", "HR-Ops", "IT-Support", "IT-Support"]  # IT-Support weighted heavier (realistic load)
CATEGORY_TO_TEAM_BIAS = {
    "access_request": ["IT-Support", "IT-Support", "Facilities"],
    "hardware_issue": ["IT-Support"],
    "software_bug": ["IT-Support"],
    "onboarding": ["HR-Ops", "Facilities", "IT-Support"],
    "hr_query": ["HR-Ops"],
}

# Deliberately uneven base resolution time per team (hours) — IT-Support is
# overloaded and slower; this is the bottleneck the analysis should find.
TEAM_BASE_RESOLUTION_HOURS = {
    "IT-Support": 30,
    "Facilities": 12,
    "HR-Ops": 18,
}

PRIORITIES = ["Low", "Medium", "High"]


@dataclass
class Ticket:
    ticket_id: int
    category: str
    description: str
    template_id: int
    team: str
    priority: str
    created_hour: float   # hours since t=0, synthetic clock
    resolution_hours: float | None
    status: str            # "Open" or "Resolved"


def generate_tickets(n=2000, seed=42, open_fraction=0.08):
    rng = random.Random(seed)
    tickets = []
    clock = 0.0
    for i in range(n):
        category = rng.choice(list(CATEGORIES.keys()))
        templates = CATEGORIES[category]
        template_idx = rng.randrange(len(templates))
        template = templates[template_idx]
        description = template.format(team=rng.choice(["Alpha", "Beta", "Gamma"]), n=rng.randint(1, 9))
        team = rng.choice(CATEGORY_TO_TEAM_BIAS[category])
        priority = rng.choices(PRIORITIES, weights=[40, 45, 15])[0]

        clock += rng.uniform(0.1, 1.5)  # tickets arrive over time
        base = TEAM_BASE_RESOLUTION_HOURS[team]
        # High priority resolved faster, some natural noise, occasional
        # long-tail delay (realistic: not every ticket resolves smoothly).
        priority_factor = {"High": 0.6, "Medium": 1.0, "Low": 1.4}[priority]
        resolution_hours = max(0.5, rng.gauss(base * priority_factor, base * 0.3))
        if rng.random() < 0.05:
            resolution_hours *= rng.uniform(2, 4)  # long-tail delayed ticket

        is_open = rng.random() < open_fraction
        tickets.append(Ticket(
            ticket_id=i + 1,
            category=category,
            description=description,
            template_id=template_idx,
            team=team,
            priority=priority,
            created_hour=round(clock, 2),
            resolution_hours=None if is_open else round(resolution_hours, 2),
            status="Open" if is_open else "Resolved",
        ))
    return tickets


def tickets_to_dicts(tickets):
    return [asdict(t) for t in tickets]


if __name__ == "__main__":
    tickets = generate_tickets()
    print(f"Generated {len(tickets)} synthetic tickets.")
    from collections import Counter
    print("Category distribution:", Counter(t.category for t in tickets))
    print("Team distribution:", Counter(t.team for t in tickets))
    print("Status distribution:", Counter(t.status for t in tickets))
