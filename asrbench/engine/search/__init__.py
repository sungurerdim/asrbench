"""
IAMS (Interaction-Aware Multi-Start) parameter optimization search package.

The package is organized as layered independent modules to keep each concern
isolated and unit-testable:

- space.py        : ParameterSpace — typed parameter declarations and YAML loading
- objective.py    : Objective — score function abstraction (single metric or weighted)
- trial.py        : TrialResult, TrialExecutor protocol + SyntheticTrialExecutor
- significance.py : CI ∧ epsilon statistical significance gate
- budget.py       : BudgetController — hard cap + convergence-based early stop
- screening.py    : Layer 1 — independent OFAT-3 sensitivity analysis
- local_1d.py     : Layers 2+6 — golden section / pattern / exhaustive 1D search
- pairwise_grid.py: Layer 3 — 3x3 grid scan on top-K sensitive pairs (interaction)
- multistart.py   : Layer 4 — multi-start coordinate descent from promising points
- ablation.py     : Layer 5 — leave-k-out ablation to detect toxic interactions
- validation.py   : Layer 7 — final config variance check and confidence certification
"""
