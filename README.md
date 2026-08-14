# Predictive-Maintenance-Cost-Aware
Cost-aware evaluation of probabilistic and ensemble models for industrial predictive maintenance — do complex models actually save money?

## Question

On near-deterministic failure boundaries, does algorithmic complexity
buy anything once models are evaluated on expected cost rather than
accuracy?

## Hypothesis

Once physics-derived features are engineered, tree-based models will
converge to near-identical performance, and the choice of alarm
threshold will affect total cost more than the choice of algorithm.

Falsified if: boosting beats a depth-limited decision tree by more
than the cross-validation spread.

## Data

- AI4I 2020 (UCI) — classification
- NASA C-MAPSS FD001 — lifetime estimation

## Status

Design phase.
