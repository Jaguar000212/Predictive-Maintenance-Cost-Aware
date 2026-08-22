"""Layer 4: cost model, threshold optimisation, and policy simulation."""

from .cost_model import (
    CVCostResult,
    cost_curve,
    cost_per_row,
    cross_validated_cost_curve,
    expected_cost,
    optimal_operating_point,
)
from .policy_sim import (
    always_alarm_counts,
    never_alarm_counts,
    oracle_ceiling_counts,
    policy_table,
)

__all__ = [
    "CVCostResult",
    "always_alarm_counts",
    "cost_curve",
    "cost_per_row",
    "cross_validated_cost_curve",
    "expected_cost",
    "never_alarm_counts",
    "optimal_operating_point",
    "oracle_ceiling_counts",
    "policy_table",
]
