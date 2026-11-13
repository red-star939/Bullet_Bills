"""Counterfactual scenario scoring with explicit non-probabilistic semantics."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RiskScore:
    scenarios: int
    exposures: int
    exposure_fraction: float


def score_future_scenarios(contact_flags):
    """Fraction of simulated contact scenarios; NOT calibrated death probability."""
    flags = [bool(x) for x in contact_flags]
    if not flags:
        raise ValueError('at least one scenario required')
    hits = sum(flags)
    return RiskScore(len(flags), hits, hits / len(flags))


def select_intervention(baseline_risk, proposed_risk, *, min_improvement=0.1,
                        absolute_limit=0.33):
    """Accept a proposal only when its scenario exposure improves sufficiently."""
    b = float(baseline_risk)
    p = float(proposed_risk)
    if not 0 <= b <= 1 or not 0 <= p <= 1:
        raise ValueError('risks must be in [0,1]')
    return p <= absolute_limit and (b-p) >= min_improvement
