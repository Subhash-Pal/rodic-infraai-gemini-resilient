"""irc_recommender.py - rule-based severity class and repair guidance.

Copied from Rodic-infra-Colab v2/irc_recommender.py (tested there on synthetic cases).

Replaces the bare "Engineering decision: not_provided" output with a severity
class, indicative repair methodology and the standards the engineer should
consult.

IMPORTANT - what this is and is not
-----------------------------------
* It is a lookup table, not a structural assessment. Every output carries
  ``needs_engineer_review = True`` and a disclaimer.
* Standards are named, not quoted: no clause numbers are cited because they
  have not been checked against the current edition. A qualified engineer must
  confirm each action against the standard text.
* The crack-width bands are STARTING VALUES taken from the jury discussion
  (0.2 mm hairline is normal, about 2 mm is dangerous). They are configurable
  (``CRACK_WIDTH_BANDS``) and must be set by the responsible engineer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

SEVERITY_ORDER = ["none", "cosmetic", "watch", "significant", "urgent"]

STANDARDS = [
    "IRC:SP:35 - Guidelines for Inspection and Maintenance of Bridges",
    "IRC:SP:40 - Guidelines on Techniques for Strengthening and Rehabilitation of Bridges",
    "MoRTH - Specifications for Road and Bridge Works",
]

DISCLAIMER = (
    "Screening aid only. This is an automated, indicative suggestion and not a "
    "structural safety judgement. A qualified engineer must validate every "
    "finding on site and confirm each action against the current IRC / MoRTH "
    "documents before any decision."
)

# (upper_bound_mm_exclusive, severity). Starting values; edit per project.
CRACK_WIDTH_BANDS = [
    (0.2, "cosmetic"),
    (0.5, "watch"),
    (2.0, "significant"),
    (float("inf"), "urgent"),
]

CRACK_ACTIONS = {
    "cosmetic": [
        "Record location and width; re-inspect at the next scheduled cycle.",
    ],
    "watch": [
        "Fix a crack monitor / tell-tale and record width over time.",
        "Seal the surface with a flexible crack sealant if exposed to moisture.",
        "Shorten the re-inspection interval for this member.",
    ],
    "significant": [
        "Clean the crack and repair by epoxy injection (epoxy grouting).",
        "Check for reinforcement corrosion along the crack.",
        "Engineer to decide whether a structural review is needed.",
    ],
    "urgent": [
        "Arrange an urgent structural audit of the member.",
        "Engineer to decide on access / load restrictions until the audit.",
        "Plan epoxy injection or strengthening only after the audit.",
    ],
}

CLASS_ALIASES = {
    "crack": "crack",
    "spallation": "spallation",
    "spalling": "spallation",
    "efflorescence": "efflorescence",
    "exposedbars": "exposed_bars",
    "exposedbar": "exposed_bars",
    "exposedrebar": "exposed_bars",
    "corrosionstain": "corrosion_stain",
    "corrosion": "corrosion_stain",
    "background": "none",
    "nodefect": "none",
    "clean": "none",
}

# Fixed severity + actions for non-crack classes (crack uses the width bands).
CLASS_RULES = {
    "spallation": (
        "significant",
        [
            "Sound the surrounding concrete and remove all loose / delaminated material.",
            "Clean the exposed surface and re-profile with polymer-modified repair mortar.",
            "Check whether reinforcement is exposed; if so also follow the exposed-bars actions.",
        ],
    ),
    "exposed_bars": (
        "urgent",
        [
            "Remove loose concrete behind the bars, remove rust and apply an anti-corrosion primer.",
            "Re-instate cover with polymer-modified mortar or micro-concrete.",
            "Engineer to check for loss of bar section; arrange a structural audit if present.",
        ],
    ),
    "corrosion_stain": (
        "watch",
        [
            "Find the moisture / chloride source and stop the ingress.",
            "Engineer to consider cover survey and corrosion NDT (e.g. half-cell potential).",
            "Apply surface protection after investigation; re-inspect at a shorter interval.",
        ],
    ),
    "efflorescence": (
        "cosmetic",
        [
            "Find and stop the water path; dry-clean the deposit.",
            "Seal or waterproof the surface; monitor for recurrence.",
        ],
    ),
}


@dataclass
class Recommendation:
    defect_class: str
    severity: str
    severity_basis: str
    actions: list = field(default_factory=list)
    references: list = field(default_factory=lambda: list(STANDARDS))
    needs_engineer_review: bool = True
    notes: list = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


def normalise_class(name: Optional[str]) -> str:
    """Map a model label to one of: crack, spallation, efflorescence,
    exposed_bars, corrosion_stain, none, unclassified."""
    if not name:
        return "unclassified"
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return CLASS_ALIASES.get(key, "unclassified")


def severity_from_crack_width(width_mm: float) -> str:
    for upper, severity in CRACK_WIDTH_BANDS:
        if width_mm < upper:
            return severity
    return CRACK_WIDTH_BANDS[-1][1]


def recommend(
    defect_class: Optional[str],
    confidence: Optional[float] = None,
    crack_width_mm: Optional[float] = None,
    width_resolvable: bool = True,
    min_confidence: float = 0.5,
) -> Recommendation:
    """Return an indicative recommendation for one finding.

    ``width_resolvable=False`` means the image cannot resolve the crack width
    (it is sub-pixel at that distance); the reported width is then only an
    upper bound and severity is marked provisional.
    """
    cls = normalise_class(defect_class)
    notes: list = []

    if confidence is not None and confidence < min_confidence:
        notes.append(
            f"Model confidence {confidence:.2f} is below {min_confidence:.2f}: "
            "treat as unconfirmed and verify visually."
        )

    if cls == "none":
        notes.append(
            "No defect above the threshold is not proof of absence: the screening models have known misses "
            "(for example cracks on plastered or painted surfaces)."
        )
        return Recommendation("none", "none", "no defect reported", [], notes=notes)

    if cls == "unclassified":
        notes.append("No usable class from the models; human inspection required.")
        return Recommendation(
            "unclassified", "watch", "provisional: unclassified",
            ["Inspect the location manually and classify the defect."], notes=notes,
        )

    if cls == "crack":
        if crack_width_mm is None:
            notes.append(
                "Crack width not measured. Measure with a crack comparator or a "
                "calibrated close-up before classifying."
            )
            return Recommendation(
                "crack", "watch", "provisional: width unknown",
                ["Measure crack width on site (crack comparator / calibrated image)."]
                + CRACK_ACTIONS["watch"],
                notes=notes,
            )
        severity = severity_from_crack_width(crack_width_mm)
        basis = f"crack width {crack_width_mm:.2f} mm"
        if not width_resolvable:
            basis = f"provisional: width <= {crack_width_mm:.2f} mm (below image resolution)"
            notes.append(
                "The crack is too thin to resolve at this distance; the width is an "
                "upper bound. Take a closer / zoomed image to measure it."
            )
        return Recommendation("crack", severity, basis, list(CRACK_ACTIONS[severity]), notes=notes)

    severity, actions = CLASS_RULES[cls]
    return Recommendation(cls, severity, f"class rule: {cls}", list(actions), notes=notes)


def worst_severity(recs: Iterable[Recommendation]) -> str:
    worst = "none"
    for rec in recs:
        if SEVERITY_ORDER.index(rec.severity) > SEVERITY_ORDER.index(worst):
            worst = rec.severity
    return worst


def severity_counts(recs: Iterable[Recommendation]) -> dict:
    counts = {s: 0 for s in SEVERITY_ORDER}
    for rec in recs:
        counts[rec.severity] += 1
    return counts
