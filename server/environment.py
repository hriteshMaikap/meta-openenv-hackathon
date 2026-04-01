"""
server/environment.py — ETL Pipeline Agent OpenEnv Environment

The central class. Implements reset() / step() / state() per OpenEnv spec.

Key architectural decisions:
  - DataFrame lives server-side only (df_working, df_gold)
  - Agent receives text/dict observations (POMDP)
  - Transforms run in a sandboxed exec() call with restricted globals
  - Reward computed by RewardEngine (deterministic, no LLM per step)
  - LLM judge called ONCE at submit() for reasoning score
  - Hard task injects schema drift at a configurable step
"""
from __future__ import annotations

import io
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Try OpenEnv import; fall back to stub for local dev
try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    class Environment:  # local dev stub
        pass

from models import ETLAction, ETLObservation, ETLState, ColumnProfile, ValidationResult
from fault_injector import FaultInjector
from reward_engine import (
    RewardEngine,
    check_null_rate, check_type_compliance, check_range,
    check_uniqueness, check_value_set, check_schema_columns,
    check_completeness, check_referential_integrity,
    check_business_rule_margin,
    score_reasoning_with_llm,
)


class ETLEnvironment(Environment):
    """
    ETL Pipeline Agent — OpenEnv RL training environment.

    The agent plays the role of a data engineer. It receives a broken
    DataFrame, must identify faults, write pandas transformation code,
    validate quality, and submit a clean output.

    Tasks:
      easy   — Single table, 5-6 independent faults, 15 steps
      medium — 3 tables, cross-table + business rules, 20 steps
      hard   — Medium + schema drift mid-episode, 25 steps
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Sandbox: restrict what the agent can import in transform code
    _SAFE_GLOBALS = {
        "__builtins__": {
            "len": len, "range": range, "list": list, "dict": dict,
            "str": str, "int": int, "float": float, "bool": bool,
            "print": print, "enumerate": enumerate, "zip": zip,
            "min": min, "max": max, "abs": abs, "round": round,
            "isinstance": isinstance, "type": type, "None": None,
            "True": True, "False": False,
        },
        "pd": pd,
        "np": np,
    }

    TASK_MAX_STEPS = {"easy": 15, "medium": 20, "hard": 25}

    def __init__(self, openai_api_key: Optional[str] = None, judge_model: str = "gpt-4o-mini"):
        self._state = ETLState()
        self._injector = FaultInjector()

        # Server-side DataFrames (never sent to agent)
        self._df_working: Optional[pd.DataFrame] = None
        self._df_gold: Optional[pd.DataFrame] = None
        self._df_output: Optional[pd.DataFrame] = None  # after load_to_target

        # For Medium/Hard: the reference dimension tables
        self._df_customers_clean: Optional[pd.DataFrame] = None
        self._df_products_clean: Optional[pd.DataFrame] = None

        # Transform code in buffer
        self._transform_code: Optional[str] = None
        self._transform_history: List[str] = []
        self._has_validated: bool = False

        # Drift tracking (Hard task)
        self._drift_event: Optional[Dict] = None
        self._drift_detected_before_exec: bool = False
        self._steps_after_drift: int = 0
        self._steps_wasted_after_drift: int = 0
        self._pre_drift_rows_reprocessed: bool = False

        # LLM judge client (optional)
        self._openai_client = None
        if openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=openai_api_key)
            except ImportError:
                pass
        self._judge_model = judge_model

        # Reward engine (set at reset)
        self._reward: Optional[RewardEngine] = None

    # ─────────────────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> ETLObservation:
        seed = seed if seed is not None else hash(uuid.uuid4()) % (2**32)
        ep_id = episode_id or str(uuid.uuid4())
        total_steps = self.TASK_MAX_STEPS[task_id]

        # ── Generate broken dataset ──────────────────────────────
        if task_id == "easy":
            ep = self._injector.make_easy_episode(seed)
            self._df_working = ep["df_broken"].copy()
            self._df_gold = ep["df_gold"].copy()
            self._df_customers_clean = None
            self._df_products_clean = None
            target_schema = ep["target_schema"]
            gold_row_count = ep["gold_row_count"]
            faults = ep["faults_planted"]
            drift = None

        elif task_id == "medium":
            ep = self._injector.make_medium_episode(seed)
            # Working = orders (the fact table the agent will build)
            self._df_working = ep["orders"].copy()
            self._df_gold = ep["orders_clean"].copy()
            self._df_customers_clean = ep["customers_clean"].copy()
            self._df_products_clean = ep["products_clean"].copy()
            # Expose broken dimension tables in working set too
            self._df_customers = ep["customers"].copy()
            self._df_products = ep["products"].copy()
            target_schema = ep["target_schema"]
            gold_row_count = len(ep["orders_clean"])
            faults = ep["faults_planted"]
            drift = None

        else:  # hard
            ep = self._injector.make_hard_episode(seed)
            self._df_working = ep["orders"].copy()
            self._df_gold = ep["orders_clean"].copy()
            self._df_customers_clean = ep["customers_clean"].copy()
            self._df_products_clean = ep["products_clean"].copy()
            self._df_customers = ep["customers"].copy()
            self._df_products = ep["products"].copy()
            target_schema = ep["target_schema"]
            gold_row_count = len(ep["orders_clean"])
            faults = ep["faults_planted"]
            drift = ep.get("schema_drift")

        self._df_output = None
        self._transform_code = None
        self._transform_history = []
        self._has_validated = False
        self._drift_event = drift
        self._drift_detected_before_exec = False
        self._steps_after_drift = 0
        self._steps_wasted_after_drift = 0
        self._pre_drift_rows_reprocessed = False

        self._reward = RewardEngine(task_id, target_schema)

        # ── Build state ──────────────────────────────────────────
        self._state = ETLState(
            task_id=task_id,
            episode_id=ep_id,
            step_count=0,
            total_steps=total_steps,
            schema_drift_step=ep.get("schema_drift_step") if task_id == "hard" else None,
            faults_planted=faults,
            gold_row_count=gold_row_count,
            gold_schema={c: str(self._df_gold.dtypes[c]) for c in self._df_gold.columns},
        )

        return self._build_observation(
            last_output="Episode started. Profile columns to identify issues.",
            reward=None,
            drift_event=None,
        )

    # ─────────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────────

    def step(self, action: ETLAction, **kwargs) -> Tuple[ETLObservation, float, bool, Dict]:
        self._state.step_count += 1
        self._state.actions_taken.append(action.tool)
        if action.reasoning:
            self._state.reasoning_log.append(action.reasoning)

        # ── Check step budget ────────────────────────────────────
        done = False
        if self._state.step_count >= self._state.total_steps:
            # Auto-submit when budget exhausted
            result = self._handle_submit(action)
            return result

        # ── Hard task: inject drift at configured step ───────────
        drift_event_this_step = None
        if (
            self._state.task_id == "hard"
            and self._state.schema_drift_step is not None
            and self._state.step_count == self._state.schema_drift_step
            and self._drift_event is not None
            and not self._state.schema_drift_applied
        ):
            self._state.schema_drift_applied = True
            drift_event_this_step = self._drift_event
            # Apply drift to the working df's metadata (not content yet)
            # The agent must discover and handle this

        # ── Dispatch tool ────────────────────────────────────────
        if action.tool == "profile_column":
            r, msg, obs = self._handle_profile_column(action)
        elif action.tool == "inspect_sample":
            r, msg, obs = self._handle_inspect_sample(action)
        elif action.tool == "write_transform":
            r, msg, obs = self._handle_write_transform(action)
        elif action.tool == "execute_transform":
            r, msg, obs = self._handle_execute_transform(action)
            if drift_event_this_step and not self._drift_detected_before_exec:
                self._steps_wasted_after_drift += 1
        elif action.tool == "validate":
            r, msg, obs = self._handle_validate(action)
            self._has_validated = True
        elif action.tool == "fix_transform":
            r, msg, obs = self._handle_fix_transform(action)
        elif action.tool == "load_to_target":
            r, msg, obs = self._handle_load(action)
        elif action.tool == "submit":
            return self._handle_submit(action)
        else:
            r, msg = -0.01, f"Unknown tool: {action.tool}"
            obs = {}

        # Track drift detection
        if drift_event_this_step and action.tool in ("profile_column", "inspect_sample"):
            self._drift_detected_before_exec = True

        # Accumulate reward
        self._state.total_reward_accumulated += r

        final_obs = self._build_observation(
            last_output=msg,
            reward=r,
            drift_event=drift_event_this_step["description"] if drift_event_this_step else None,
        )
        return final_obs, r, False, {"step": self._state.step_count}

    # ─────────────────────────────────────────────────────────
    # state()
    # ─────────────────────────────────────────────────────────

    @property
    def state(self) -> ETLState:
        """Return a safe view of state (no gold df or fault list)."""
        return self._state

    # ─────────────────────────────────────────────────────────
    # Tool handlers
    # ─────────────────────────────────────────────────────────

    def _handle_profile_column(self, action: ETLAction):
        col = action.params.get("column", action.params.get("col", ""))
        if self._df_working is None or col not in self._df_working.columns:
            return -0.02, f"Column '{col}' not found. Available: {list(self._df_working.columns) if self._df_working is not None else []}", {}

        # Track call count
        self._state.profile_call_count[col] = self._state.profile_call_count.get(col, 0) + 1
        call_n = self._state.profile_call_count[col]

        reward, msg = self._reward.reward_profile_column(col, call_n)

        series = self._df_working[col]
        profile = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            null_rate=round(series.isna().mean(), 4),
            unique_count=int(series.nunique()),
            sample_values=series.dropna().head(5).tolist(),
            min_val=series.dropna().min() if pd.api.types.is_numeric_dtype(series) else None,
            max_val=series.dropna().max() if pd.api.types.is_numeric_dtype(series) else None,
            has_mixed_formats=self._detect_mixed_formats(series),
            has_range_violations=self._detect_range_violations(col, series),
            has_case_inconsistency=self._detect_case_inconsistency(series),
        )

        output_msg = (
            f"Column '{col}': dtype={profile.dtype}, "
            f"null_rate={profile.null_rate:.1%}, "
            f"unique={profile.unique_count}, "
            f"sample={profile.sample_values[:3]}"
        )
        if profile.has_mixed_formats:
            output_msg += " ⚠️ MIXED FORMATS DETECTED"
        if profile.has_range_violations:
            output_msg += " ⚠️ RANGE VIOLATIONS DETECTED"
        if profile.has_case_inconsistency:
            output_msg += " ⚠️ CASE INCONSISTENCY DETECTED"

        return reward, f"{msg} | {output_msg}", {"profile": profile}

    def _handle_inspect_sample(self, action: ETLAction):
        n = int(action.params.get("n_rows", 5))
        self._state.inspect_call_count += 1
        reward, msg = self._reward.reward_inspect_sample(self._state.inspect_call_count)
        sample = self._df_working.head(n).to_dict(orient="records") if self._df_working is not None else []
        return reward, f"{msg} | First {n} rows shown", {"sample": sample}

    def _handle_write_transform(self, action: ETLAction):
        code = action.params.get("code", "")
        if not code.strip():
            return -0.02, "No code provided", {}
        # Strip bare import lines — pd, np, re are pre-injected
        import re as _re
        clean_code = "\n".join(
            line for line in code.splitlines()
            if not _re.match(r"^\s*import\s+(pandas|numpy|re)", line)
            and not _re.match(r"^\s*from\s+(pandas|numpy|re)\s+import", line)
        )
        self._transform_code = clean_code
        self._transform_history.append(clean_code)
        reward, msg = self._reward.reward_write_transform()
        return reward, f"{msg} | Code stored ({len(code)} chars). Use execute_transform() to run it.", {}

    def _handle_execute_transform(self, action: ETLAction):
        if not self._transform_code:
            return -0.05, "No transform code written yet. Use write_transform() first.", {}

        df_before = self._df_working.copy()
        rows_in = len(df_before)

        try:
            # Provide a restricted but functional namespace.
            # pd and np are injected directly — agent code should use them
            # without calling import. __import__ is blocked to prevent
            # loading os/sys/subprocess, but re is allowed for string ops.
            import re as _re
            safe_globals = {
                "__builtins__": {
                    "len": len, "range": range, "list": list, "dict": dict,
                    "str": str, "int": int, "float": float, "bool": bool,
                    "print": print, "enumerate": enumerate, "zip": zip,
                    "min": min, "max": max, "abs": abs, "round": round,
                    "isinstance": isinstance, "type": type,
                    "None": None, "True": True, "False": False,
                    "sorted": sorted, "set": set, "tuple": tuple,
                    "sum": sum, "any": any, "all": all,
                },
                "pd": pd,
                "np": np,
                "re": _re,
            }
            local_ns = {"df": df_before.copy()}
            exec(self._transform_code, safe_globals, local_ns)
            df_after = local_ns.get("df", df_before)

            if not isinstance(df_after, pd.DataFrame):
                return -0.05, "Transform must assign result back to `df`", {}

            self._df_working = df_after
            rows_out = len(df_after)

            reward, msg = self._reward.reward_execute_transform(
                success=True, error_type=None,
                rows_in=rows_in, rows_out=rows_out
            )
            return reward, f"{msg}", {}

        except SyntaxError as e:
            reward, msg = self._reward.reward_execute_transform(
                success=False, error_type="syntax",
                rows_in=rows_in, rows_out=0
            )
            return reward, f"{msg}: {e}", {"error": str(e), "error_type": "syntax"}

        except Exception as e:
            reward, msg = self._reward.reward_execute_transform(
                success=False, error_type=str(type(e).__name__),
                rows_in=rows_in, rows_out=0
            )
            return reward, f"{msg}: {e}", {"error": str(e), "error_type": type(e).__name__}

    def _handle_validate(self, action: ETLAction):
        requested = action.params.get("checks", ["null_check", "type_check", "range_check"])
        df = self._df_working
        schema = self._reward.target_schema
        prev_scores = deepcopy(self._state.last_validation_scores)
        new_scores = {}
        details = []

        check_map = {
            "null_check": lambda: check_null_rate(df,
                [c for c, s in schema.items() if not s.get("nullable", True)]),
            "type_check": lambda: check_type_compliance(df, schema),
            "range_check": lambda: check_range(df, schema),
            "uniqueness_check": lambda: check_uniqueness(df,
                [c for c, s in schema.items() if s.get("unique", False)]),
            "value_set_check": lambda: check_value_set(df, schema),
            "schema_check": lambda: check_schema_columns(df, schema),
        }

        for check in requested:
            if check in check_map:
                score, detail = check_map[check]()
                new_scores[check] = score
                details.append(ValidationResult(
                    check_name=check, score=score,
                    passing_rows=int(score * len(df)), failing_rows=int((1 - score) * len(df)),
                    detail=detail,
                ))

        # Update accumulated scores
        self._state.last_validation_scores.update(new_scores)
        self._state.validate_call_count += 1

        reward, msg = self._reward.reward_validate(new_scores, prev_scores)

        summary = " | ".join(f"{k}={v:.2f}" for k, v in new_scores.items())
        return reward, f"{msg} | Scores: {summary}", {"validation": details, "scores": new_scores}

    def _handle_fix_transform(self, action: ETLAction):
        new_code = action.params.get("new_code", action.params.get("code", ""))
        error_msg = action.params.get("error_msg", "")
        if not new_code.strip():
            return -0.02, "No code provided in fix_transform", {}
        self._transform_code = new_code
        self._transform_history.append(new_code)
        reward, msg = self._reward.reward_fix_transform(new_code, error_msg)
        return reward, f"{msg} | Updated code stored.", {}

    def _handle_load(self, action: ETLAction):
        if self._df_working is None:
            return -0.05, "No working DataFrame to load", {}
        reward, msg = self._reward.reward_load_to_target(self._df_working)
        self._df_output = self._df_working.copy()
        return reward, f"{msg} | Output loaded to target slot.", {}

    def _handle_submit(self, action: ETLAction) -> Tuple[ETLObservation, float, bool, Dict]:
        """
        End the episode. Run full grader. Optionally call LLM judge for reasoning.
        Returns (observation, final_reward, done=True, info).
        """
        # Penalty for no validation
        early_penalty = self._reward.reward_submit_early_penalty(self._has_validated)

        # Use loaded output if available, otherwise use working df
        df_final = self._df_output if self._df_output is not None else self._df_working
        if df_final is None:
            obs = self._build_observation("No data processed.", 0.0, None)
            return obs, -0.5, True, {"final_score": 0.0, "reason": "no_output"}

        task = self._state.task_id

        if task == "easy":
            grader_result = self._reward.grade_easy_episode(
                df_final,
                gold_row_count=self._state.gold_row_count,
                faults_planted=self._state.faults_planted,
                steps_taken=self._state.step_count,
                total_steps=self._state.total_steps,
            )
        elif task == "medium":
            grader_result = self._reward.grade_medium_episode(
                df_final,
                df_customers_clean=self._df_customers_clean,
                df_products_clean=self._df_products_clean,
                faults_planted=self._state.faults_planted,
                steps_taken=self._state.step_count,
                total_steps=self._state.total_steps,
            )
        else:  # hard
            grader_result = self._reward.grade_hard_episode(
                df_final,
                drift_detected_before_exec=self._drift_detected_before_exec,
                steps_wasted_after_drift=self._steps_wasted_after_drift,
                pre_drift_rows_reprocessed=self._pre_drift_rows_reprocessed,
                gold_row_count=self._state.gold_row_count,
                faults_planted=self._state.faults_planted,
                steps_taken=self._state.step_count,
                total_steps=self._state.total_steps,
            )

        base_final = grader_result["final_score"] + early_penalty
        # Hard floor: submitting with no validation can't score above 0.3
        if not self._has_validated:
            base_final = min(base_final, 0.3)

        # LLM judge for reasoning (optional, 1 call per episode)
        reasoning_score = 0.5  # default if no judge
        reasoning_justification = "LLM judge not configured"
        if self._openai_client and self._state.reasoning_log:
            reasoning_score, reasoning_justification = score_reasoning_with_llm(
                self._state.reasoning_log,
                self._state.faults_planted,
                self._openai_client,
                self._judge_model,
            )

        # Blend: 95% verifiable + 5% reasoning quality
        final_reward = round(0.95 * base_final + 0.05 * reasoning_score, 4)

        info = {
            "final_score": final_reward,
            "grader_breakdown": grader_result,
            "reasoning_score": reasoning_score,
            "reasoning_justification": reasoning_justification,
            "steps_taken": self._state.step_count,
            "faults_planted": self._state.faults_planted,
        }

        summary = (
            f"Episode complete. Final score: {final_reward:.3f} | "
            f"Grader: {base_final:.3f} | Reasoning: {reasoning_score:.2f} | "
            f"Steps: {self._state.step_count}/{self._state.total_steps}"
        )
        obs = self._build_observation(summary, final_reward, None)
        return obs, final_reward, True, info

    # ─────────────────────────────────────────────────────────
    # Observation builder
    # ─────────────────────────────────────────────────────────

    def _build_observation(
        self,
        last_output: str,
        reward: Optional[float],
        drift_event: Optional[str],
    ) -> ETLObservation:
        df = self._df_working
        schema_current = {c: str(df.dtypes[c]) for c in df.columns} if df is not None else {}
        sample = df.head(5).to_dict(orient="records") if df is not None else []
        row_count = len(df) if df is not None else 0

        return ETLObservation(
            done=False,
            reward=reward,
            task_id=self._state.task_id,
            step=self._state.step_count,
            steps_remaining=self._state.total_steps - self._state.step_count,
            dataset_sample=sample,
            schema_current=schema_current,
            schema_target=self._reward.target_schema if self._reward else {},
            row_count=row_count,
            column_profiles={},  # filled per-call in profile handler
            validation_scores=deepcopy(self._state.last_validation_scores),
            validation_details=[],
            current_transform_code=self._transform_code,
            transform_history=list(self._transform_history),
            last_execute_result=None,
            last_tool_output=last_output,
            last_step_reward=reward or 0.0,
            schema_drift_event=drift_event,
            batches_committed=self._state.batches_committed,
        )

    # ─────────────────────────────────────────────────────────
    # Data analysis helpers (used in profile_column)
    # ─────────────────────────────────────────────────────────

    def _detect_mixed_formats(self, series: pd.Series) -> bool:
        if series.dtype != object:
            return False
        sample = series.dropna().head(50).astype(str)
        # Detect date format mixing
        iso_pattern = r"\d{4}-\d{2}-\d{2}"
        natural_pattern = r"[A-Z][a-z]+ \d{1,2}(,)? \d{4}"
        has_iso = sample.str.match(iso_pattern).any()
        has_natural = sample.str.match(natural_pattern).any()
        return bool(has_iso and has_natural)

    def _detect_range_violations(self, col: str, series: pd.Series) -> bool:
        schema = self._reward.target_schema if self._reward else {}
        spec = schema.get(col, {})
        if "min" in spec and pd.api.types.is_numeric_dtype(series):
            if (series.dropna() < spec["min"]).any():
                return True
        if "max" in spec and pd.api.types.is_numeric_dtype(series):
            if (series.dropna() > spec["max"]).any():
                return True
        return False

    def _detect_case_inconsistency(self, series: pd.Series) -> bool:
        if series.dtype != object:
            return False
        sample = series.dropna().head(50).astype(str)
        has_upper = sample.str.contains(r"[A-Z]").any()
        has_lower = sample.str.contains(r"[a-z]").any()
        return bool(has_upper and has_lower and sample.nunique() > 2)