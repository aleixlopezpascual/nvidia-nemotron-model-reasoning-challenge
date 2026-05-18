"""6-way puzzle-type classifier.

All training prompts begin with one of six deterministic "In Alice's Wonderland..."
phrases (verified empirically on train.csv — total 9,500 rows split ~1,550-1,600 each).
Classification is exact prefix matching on the prompt's leading clause.
"""

from __future__ import annotations

PUZZLE_TYPES: tuple[str, ...] = (
    "bit_manipulation",
    "gravitational_constant",
    "unit_conversion",
    "text_encryption",
    "numeral_conversion",
    "equation_transformation",
)

# Leading-clause → label. The keys are the longest unique prefix needed for an
# unambiguous match against the empirically observed prompt leads.
_LEAD_TO_LABEL: tuple[tuple[str, str], ...] = (
    ("In Alice's Wonderland, a secret bit manipulation rule", "bit_manipulation"),
    ("In Alice's Wonderland, the gravitational constant", "gravitational_constant"),
    ("In Alice's Wonderland, a secret unit conversion", "unit_conversion"),
    ("In Alice's Wonderland, secret encryption rules", "text_encryption"),
    ("In Alice's Wonderland, numbers are secretly converted", "numeral_conversion"),
    ("In Alice's Wonderland, a secret set of transformation rules is applied to equations", "equation_transformation"),
)


def classify_puzzle_type(prompt: str) -> str:
    """Return one of PUZZLE_TYPES, or 'unknown' if no lead matches."""
    for lead, label in _LEAD_TO_LABEL:
        if prompt.startswith(lead):
            return label
    return "unknown"
