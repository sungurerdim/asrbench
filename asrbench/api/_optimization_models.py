"""Pydantic request / response models for the /optimize API.

Split out of ``optimization.py`` to keep that module focused on the HTTP
surface. Moving schema classes here has no runtime effect — the wire
format is identical.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "BudgetConfig",
    "GlobalConfigStartRequest",
    "GlobalConfigStartResponse",
    "GlobalDatasetSpec",
    "ObjectiveConfig",
    "OptimizeStartRequest",
    "OptimizeStartResponse",
    "StudyResponse",
    "TrialResponse",
    "TwoStageStartRequest",
    "TwoStageStartResponse",
]


class ObjectiveConfig(BaseModel):
    """
    Objective function payload.

    - For a single metric: {"type": "single", "metric": "wer",
                            "direction": "minimize" | "maximize" | null}
    - For weighted: {"type": "weighted", "weights": {"wer": 1.0, "rtfx": -0.1}}
    """

    type: str = Field(..., pattern="^(single|weighted)$")
    metric: str | None = None
    direction: str | None = None
    weights: dict[str, float] | None = None


class BudgetConfig(BaseModel):
    hard_cap: int = Field(..., gt=0)
    convergence_eps: float = Field(0.005, ge=0)
    convergence_window: int = Field(3, ge=0)


class OptimizeStartRequest(BaseModel):
    model_id: str
    dataset_id: str
    lang: str = "en"
    space: dict[str, Any]
    objective: ObjectiveConfig
    mode: str = Field("maximum", pattern="^(fast|balanced|maximum)$")
    budget: BudgetConfig
    eps_min: float = 0.005
    top_k_pairs: int = Field(4, ge=2)
    multistart_candidates: int = Field(3, ge=1)
    validation_runs: int = Field(3, ge=2)
    enable_deep_ablation: bool = False
    prior_study_id: str | None = Field(
        None, description="Resume from a completed study's cache + screening"
    )


class OptimizeStartResponse(BaseModel):
    study_id: str
    status: str
    mode: str
    hard_cap: int


class TwoStageStartRequest(BaseModel):
    """
    Kick off a two-stage coarse→fine IAMS run.

    The request carries the same fields as ``OptimizeStartRequest`` PLUS
    two durations. Budget/epsilon are optional — if omitted the library's
    auto-sizing helpers size them from the space and stage durations.
    """

    model_id: str
    dataset_id: str
    lang: str = "en"
    space: dict[str, Any]
    objective: ObjectiveConfig
    mode: str = Field("maximum", pattern="^(fast|balanced|maximum)$")
    stage1_duration_s: int = Field(900, gt=0)
    stage2_duration_s: int = Field(2400, gt=0)
    stage1_budget: int | None = Field(None, ge=1)
    stage2_budget: int | None = Field(None, ge=1)
    stage1_epsilon: float | None = Field(None, ge=0)
    stage2_epsilon: float | None = Field(None, ge=0)
    top_k_pairs: int = Field(4, ge=2)
    multistart_candidates: int = Field(3, ge=1)
    validation_runs: int = Field(3, ge=2)
    enable_deep_ablation: bool = False
    use_multifidelity: bool = Field(
        False,
        description=(
            "Enable Hyperband-style rung pruning for Layer 2+ trials. "
            "Cheap configs get evaluated at 25%/50%/100% of the corpus; "
            "clearly-worse partial scores short-circuit the trial. "
            "Layer 1 screening and Layer 7 validation stay at full fidelity."
        ),
    )


class TwoStageStartResponse(BaseModel):
    stage1_study_id: str
    stage2_study_id: str
    status: str
    mode: str


class GlobalDatasetSpec(BaseModel):
    """One dataset slot in a global-config run."""

    dataset_id: str
    lang: str = "en"
    weight: float = Field(1.0, gt=0)


class GlobalConfigStartRequest(BaseModel):
    """
    Kick off a two-stage IAMS run over N datasets simultaneously.

    All datasets are evaluated by every IAMS trial; their scores are combined
    via ``MultiDatasetTrialExecutor`` (variance-weighted CI, weighted mean
    score) so the optimizer produces ONE config that minimizes the aggregate
    objective across the whole fleet. Use this when deploying to a product
    with a single shared preset.
    """

    model_id: str
    datasets: list[GlobalDatasetSpec] = Field(..., min_length=1)
    space: dict[str, Any]
    objective: ObjectiveConfig
    mode: str = Field("maximum", pattern="^(fast|balanced|maximum)$")
    stage1_duration_s: int = Field(900, gt=0)
    stage2_duration_s: int = Field(2400, gt=0)
    stage1_budget: int | None = Field(None, ge=1)
    stage2_budget: int | None = Field(None, ge=1)
    stage1_epsilon: float | None = Field(None, ge=0)
    stage2_epsilon: float | None = Field(None, ge=0)
    top_k_pairs: int = Field(4, ge=2)
    multistart_candidates: int = Field(3, ge=1)
    validation_runs: int = Field(3, ge=2)
    enable_deep_ablation: bool = False
    use_multifidelity: bool = False


class GlobalConfigStartResponse(BaseModel):
    stage1_study_id: str
    stage2_study_id: str
    status: str
    mode: str
    dataset_count: int


class TrialResponse(BaseModel):
    trial_id: str
    run_id: str | None
    phase: str
    config: dict
    score: float | None
    score_ci_lower: float | None
    score_ci_upper: float | None
    reasoning: str | None
    created_at: str


class StudyResponse(BaseModel):
    study_id: str
    model_id: str
    dataset_id: str
    lang: str
    mode: str
    status: str
    eps_min: float
    best_run_id: str | None
    best_score: float | None
    best_config: dict | None
    confidence: str | None
    total_trials: int | None
    reasoning: list[str] | None
    started_at: str | None
    finished_at: str | None
    created_at: str
    error_message: str | None = None
