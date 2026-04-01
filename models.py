"""
models.py — Typed contracts for the ETL Pipeline Agent OpenEnv environment.

Design philosophy (grounded in research):
  - All fields are Pydantic v2 models (required by OpenEnv spec)
  - Observation is POMDP: agent sees telemetry, NOT the gold dataset
  - Action is a tagged union over 8 discrete tools (no free-form shell)
  - Reward is verifiable: pandas computes it, not an LLM judge
  - LLM judge used ONLY once per episode (submit reasoning scoring)

MDP classification:
  - Type: Finite-horizon POMDP
  - State space: structured tabular (DataFrame) + history
  - Action space: discrete structured (8 tools with typed params)
  - Reward: dense multi-objective (per-step + episode-final)
  - Horizon: 15 / 20 / 25 steps by task
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


# ─────────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────────

ToolName = Literal[
    "profile_column",       # stat profile of one column
    "inspect_sample",       # show N rows
    "write_transform",      # store pandas code (no exec yet)
    "execute_transform",    # run stored code against live df
    "validate",             # run quality checks, get float scores
    "fix_transform",        # revise stored code after error
    "load_to_target",       # write output df to target slot
    "submit",               # end episode, trigger full grader
]


class ETLAction(Action):
    """
    One agent step. The LLM outputs this as structured JSON.

    Design notes:
      - `tool` is strictly enumerated — prevents free-form shell injection
      - `params` is a typed dict per tool (validated in env.step())
      - `reasoning` is logged every step and scored ONCE at submit()
        using a lightweight LLM rubric call (not per-step, low cost)
    """
    tool: ToolName = Field(..., description="Which tool to invoke")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Tool-specific parameters. Examples:\n"
            "  profile_column:   {'column': 'order_date'}\n"
            "  inspect_sample:   {'n_rows': 5}\n"
            "  write_transform:  {'code': 'df[...] = ...'}\n"
            "  execute_transform:{}\n"
            "  validate:         {'checks': ['null_check', 'type_check']}\n"
            "  fix_transform:    {'error_msg': '...', 'new_code': '...'}\n"
            "  load_to_target:   {}\n"
            "  submit:           {'reasoning': 'I fixed X by doing Y because Z'}"
        ),
    )
    reasoning: str = Field(
        default="",
        description=(
            "Agent's chain-of-thought for this step. "
            "Scored at episode end via LLM rubric (not per-step). "
            "Logged for interpretability regardless."
        ),
    )


# ─────────────────────────────────────────────
# OBSERVATION  (what the agent sees — POMDP)
# ─────────────────────────────────────────────

class ColumnProfile(State):
    """Statistics for one column, revealed by profile_column()."""
    name: str
    dtype: str
    null_rate: float = Field(ge=0.0, le=1.0)
    unique_count: int
    sample_values: List[Any]
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    # Fault hints (what the env reveals after profiling)
    has_mixed_formats: bool = False
    has_range_violations: bool = False
    has_case_inconsistency: bool = False


class ValidationResult(State):
    """Score for one quality check (0.0–1.0)."""
    check_name: str
    score: float = Field(ge=0.0, le=1.0)
    passing_rows: int
    failing_rows: int
    detail: str  # e.g. "3 rows have order_id duplicated: [1002, 1002]"


class ETLObservation(Observation):
    """
    What the agent sees after every step.

    The gold-standard clean DataFrame is NEVER in here — it lives
    server-side only in ETLState. The agent must infer what needs
    fixing from the profiles and validation results it collects.

    This is the POMDP observation: informative but incomplete.
    """
    # ── Always visible ──────────────────────────────────────────
    task_id: str = Field(description="'easy' | 'medium' | 'hard'")
    step: int = Field(description="Current step index (0-based)")
    steps_remaining: int

    # ── Dataset telemetry ───────────────────────────────────────
    dataset_sample: List[Dict[str, Any]] = Field(
        description="First 5 rows of current df as dicts (after any transforms)"
    )
    schema_current: Dict[str, str] = Field(
        description="Column → inferred dtype of current working df"
    )
    schema_target: Dict[str, Any] = Field(
        description="The contract the output must satisfy"
    )
    row_count: int = Field(description="Current row count of working df")

    # ── Revealed by actions (accumulate over episode) ───────────
    column_profiles: Dict[str, ColumnProfile] = Field(
        default_factory=dict,
        description="Column profiles revealed by profile_column() calls"
    )
    validation_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Latest score per quality check (0.0–1.0)"
    )
    validation_details: List[ValidationResult] = Field(
        default_factory=list,
        description="Full detail from last validate() call"
    )

    # ── Transform state ─────────────────────────────────────────
    current_transform_code: Optional[str] = Field(
        default=None, description="Code currently stored (not yet exec'd if just written)"
    )
    transform_history: List[str] = Field(
        default_factory=list,
        description="All transform code versions tried this episode"
    )
    last_execute_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Result of last execute_transform(): rows_in, rows_out, errors"
    )

    # ── Step feedback ────────────────────────────────────────────
    last_tool_output: str = Field(
        default="",
        description="Human-readable output of the last tool call"
    )
    last_step_reward: float = Field(
        default=0.0,
        description="Reward signal from the last step"
    )

    # ── Hard task only ──────────────────────────────────────────
    schema_drift_event: Optional[str] = Field(
        default=None,
        description="Non-None only when schema drift is injected (Hard task)"
    )
    batches_committed: int = Field(
        default=0,
        description="For Hard task: how many batches already loaded (do not reprocess)"
    )


# ─────────────────────────────────────────────
# STATE  (server-side — never sent to agent)
# ─────────────────────────────────────────────

class ETLState(State):
    """
    Full server-side episode state.

    Hidden fields (df_gold, faults_planted) are NEVER serialised
    to the agent. state() exposes only the safe subset.
    """
    task_id: str = "easy"
    episode_id: str = ""

    # ── Safe to expose via state() ──────────────────────────────
    step_count: int = 0
    total_steps: int = 15          # 15 / 20 / 25
    loaded_to_target: bool = False

    # Hard task specific
    schema_drift_step: Optional[int] = None   # step at which drift fires
    schema_drift_applied: bool = False
    batches_committed: int = 0

    # ── Hidden (grader only) ────────────────────────────────────
    # These are set at reset() and compared at grade_episode()
    # They are excluded from any serialisation to the agent.
    faults_planted: List[str] = Field(
        default_factory=list, exclude=True
    )
    gold_row_count: int = Field(default=0, exclude=True)
    gold_schema: Dict[str, str] = Field(default_factory=dict, exclude=True)

    # ── Accumulator for final scoring ──────────────────────────
    reasoning_log: List[str] = Field(
        default_factory=list,
        description="All reasoning strings collected (scored at submit)"
    )
    total_reward_accumulated: float = 0.0
    actions_taken: List[str] = Field(default_factory=list)

    # ── Reward hacking guards ───────────────────────────────────
    profile_call_count: Dict[str, int] = Field(default_factory=dict)
    inspect_call_count: int = 0
    validate_call_count: int = 0
    last_validation_scores: Dict[str, float] = Field(default_factory=dict)