from __future__ import annotations

import copy
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Literal, Optional


StageName = Literal["remember", "recall", "apply", "guard"]
RewardTrend = Literal["improving", "stable", "degrading"]
PrimaryMetric = Literal["accuracy", "harmlessness", "pass_rate"]

STAGES: tuple[StageName, ...] = ("remember", "recall", "apply", "guard")


@dataclass
class CaseFeatures:
    input_length: int
    complexity_estimate: float
    domain: str
    requires_external_knowledge: bool
    safety_sensitivity: float
    ambiguity_score: float
    instruction: str = ""
    cluster_id: Optional[str] = None


@dataclass
class BenchmarkCase:
    case_id: str
    prompt: str
    expected_output: str
    features: CaseFeatures
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessConfig:
    """The frozen compatible base plan, grouped by personalization stage.

    Each field is the base the matching ``harness.py`` hook patches over, so a
    knob lives with the decision that consumes it: what this request may see of
    the user is ``recall``, how to answer for them is ``apply``, and how the
    answer is checked is ``guard``. ``remember`` has no upstream knob because
    the memory compiler's default is engine-owned.
    """

    remember: dict[str, Any] = field(default_factory=dict)
    recall: dict[str, Any] = field(default_factory=dict)
    apply: dict[str, Any] = field(default_factory=dict)
    guard: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "HarnessConfig":
        return HarnessConfig(
            remember=copy.deepcopy(self.remember),
            recall=copy.deepcopy(self.recall),
            apply=copy.deepcopy(self.apply),
            guard=copy.deepcopy(self.guard),
        )

    def update_stage(self, stage: StageName, updates: dict[str, Any]) -> None:
        current = getattr(self, stage)
        current.update(updates)

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {stage: dict(getattr(self, stage)) for stage in STAGES}

    def delta_from(self, previous: "HarnessConfig") -> dict[StageName, dict[str, Any]]:
        delta: dict[StageName, dict[str, Any]] = {}
        for dim in STAGES:
            current = getattr(self, dim)
            prior = getattr(previous, dim)
            dim_delta = {
                key: value
                for key, value in current.items()
                if key not in prior or prior[key] != value
            }
            removed_keys = [key for key in prior if key not in current]
            if removed_keys:
                dim_delta["_removed"] = removed_keys
            if dim_delta:
                delta[dim] = dim_delta
        return delta


@dataclass
class Trajectory:
    num_llm_calls: int
    total_tokens: int
    latency_ms: int
    tools_invoked: list[str]
    intermediate_outputs: list[str]
    final_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    trajectory: Trajectory
    final_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticSignal:
    primary_stage: StageName
    secondary_stages: list[StageName] = field(default_factory=list)


@dataclass
class Diagnosis:
    success: bool
    analysis: str
    diagnostic_signal: DiagnosticSignal


@dataclass
class PerCaseEntry:
    case_id: str
    iteration: int
    case_features: CaseFeatures
    config: HarnessConfig
    delta_from_prev: dict[StageName, dict[str, Any]]
    trajectory: Trajectory
    primary_reward: float
    cost_actual: float
    diagnosis: Diagnosis
    # Candidate observations are separate from deployment. A rejected proposal
    # remains useful evidence and must retain its actual search parent.
    parent_iteration: int | None = None
    aggregation_weight: float | None = None
    aggregation_group: str = ""
    source_group: str = ""
    reflection_eligible: bool = True


@dataclass
class CaseHistoryRecord:
    iteration: int
    success: bool
    primary_reward: float


@dataclass
class CaseStats:
    history: list[CaseHistoryRecord] = field(default_factory=list)
    consecutive_failures: int = 0
    avg_reward_recent_N: float = 0.0
    reward_trend: RewardTrend = "stable"
    stage_failure_counts: dict[StageName, int] = field(
        default_factory=lambda: {stage: 0 for stage in STAGES}
    )

    def update(self, entry: PerCaseEntry, recent_window: int = 5) -> None:
        success = entry.diagnosis.success
        self.history.append(
            CaseHistoryRecord(
                iteration=entry.iteration,
                success=success,
                primary_reward=entry.primary_reward,
            )
        )
        self.consecutive_failures = 0 if success else self.consecutive_failures + 1
        if not success:
            primary_stage = entry.diagnosis.diagnostic_signal.primary_stage
            self.stage_failure_counts[primary_stage] += 1
            for dim in entry.diagnosis.diagnostic_signal.secondary_stages:
                self.stage_failure_counts[dim] += 1

        recent = self.history[-recent_window:]
        self.avg_reward_recent_N = mean(record.primary_reward for record in recent)
        if len(recent) < 2:
            self.reward_trend = "stable"
            return

        deltas = [
            recent[index].primary_reward - recent[index - 1].primary_reward
            for index in range(1, len(recent))
        ]
        avg_delta = mean(deltas)
        if avg_delta > 0.05:
            self.reward_trend = "improving"
        elif avg_delta < -0.05:
            self.reward_trend = "degrading"
        else:
            self.reward_trend = "stable"


@dataclass
class GlobalPattern:
    pattern_id: str
    iteration_range: tuple[int, int]
    description: str
    evidence: list[str]
    effect: str
    primary_stage: StageName = "apply"  # 主要锚定的失败阶段


@dataclass
class FeatureFilter:
    field: str
    operator: Literal["eq", "gt", "gte", "lt", "lte", "in"]
    value: Any


@dataclass
class RetrievalRequest:
    feature_filters: list[FeatureFilter] = field(default_factory=list)
    min_consecutive_failures: Optional[int] = None
    reward_trend: Optional[RewardTrend] = None
    primary_stage: Optional[StageName] = None
    iteration_range: Optional[tuple[int, int]] = None
    case_ids: Optional[list[str]] = None
    sample_k: Optional[int] = None
    sample_by_cluster_k: Optional[int] = None
    include_global_patterns: bool = True
    max_entries: Optional[int] = None
    max_global_patterns: Optional[int] = None


@dataclass
class RetrievedSlice:
    global_patterns: list[GlobalPattern]
    entries: list[PerCaseEntry]
    case_stats: dict[str, CaseStats]


@dataclass
class RewardConfig:
    primary_metric: PrimaryMetric = "accuracy"


@dataclass
class CostConstraint:
    max_tokens_per_case: int
    max_latency_ms: int
    max_llm_calls_per_case: int

    def allows(self, trajectory: Trajectory) -> bool:
        return (
            trajectory.total_tokens <= self.max_tokens_per_case
            and trajectory.latency_ms <= self.max_latency_ms
            and trajectory.num_llm_calls <= self.max_llm_calls_per_case
        )


@dataclass
class ControllerDecision:
    config: HarnessConfig
    rationale: str
    variants: list[HarnessConfig] = field(default_factory=list)


@dataclass
class CandidateRecord:
    iteration: int
    config: HarnessConfig
    primary_reward: float
    cost_actual: float
    constraint_satisfied: bool
    rationale: str


@dataclass
class TrainingReport:
    best_candidate: Optional[CandidateRecord]
    candidates: list[CandidateRecord]
    experience_bank_size: int
    global_patterns: list[GlobalPattern]


@dataclass
class EvaluationCaseResult:
    case_id: str
    config: HarnessConfig
    primary_reward: float
    trajectory: Trajectory
    diagnosis: Diagnosis


@dataclass
class EvaluationReport:
    base_config: HarnessConfig
    results: list[EvaluationCaseResult]


def make_minimal_config() -> HarnessConfig:
    return HarnessConfig(
        remember={},
        recall={
            "retrieval_mode": "none",
            "top_k": 0,
            "memory_policy": "disabled",
            "history_keep": 0,
        },
        apply={
            "structured_instruction": False,
            "compression": "none",
            "tool_access": "disabled",
            "tool_protocol": "native",
            "temperature": 0.0,
            "max_tokens": 256,
            "top_p": 1.0,
            "candidate_count": 1,
            "workflow": "single_call",
            "stop_rule": "immediate",
        },
        guard={
            "postprocess": "raw_passthrough",
            "validator": "disabled",
            "fallback": "none",
        },
    )
