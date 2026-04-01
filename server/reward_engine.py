"""
server/reward_engine.py — Multi-objective reward computation for ETL agent.

REWARD DESIGN PHILOSOPHY (research-grounded):

1. VERIFIABLE REWARDS ONLY for per-step and final grader
   Source: GRPO / RLVR literature (DeepSeek-R1, Cameron Wolfe 2025)
   Reason: LLM judges introduce reward hacking surface and 8× API cost
   during GRPO rollouts. Pandas can compute null rates, type compliance,
   row counts, schema match — all in <10ms, exactly, reproducibly.

2. LLM JUDGE for ONE signal only: reasoning quality at submit()
   Source: MT-GRPO (arXiv 2505.11821) — "turn-level LLM-as-judge
   enables more flexible and nuanced evaluation" as complement to
   verifiable rewards. We use it ONCE per episode, not per-step.
   Cost: 1 LLM call per episode vs 8*N per step.
   Benefit: teaches the model to produce correct, grounded reasoning.

3. REWARD HACKING PREVENTION
   Source: GRPO literature — "models learn to exploit flaws in the
   reward signal rather than producing genuinely better completions"
   Mitigations:
     - Diminishing returns on repeat profile calls
     - No reward for validate() if scores didn't improve
     - Step penalty scales with wasted steps (efficiency signal)
     - Reward cliff for submit() without prior validation

4. DENSE SIGNAL DESIGN
   Source: MT-GRPO paper finding: "integrating turn-level rewards
   enables RL algorithms to significantly outperform baseline methods
   with trajectory-level rewards"
   Every step gives a signal. Even write_transform() gives 0.0 not
   silence — the model knows the action was registered.

5. REWARD TENSIONS (what makes the grader non-trivial)
   - Drop all rows → perfect null_check, terrible completeness_score
   - Keep all rows → high completeness, fails referential_integrity
   - Guess-and-submit early → low final score + efficiency penalty
   These tensions prevent trivial solutions and force genuine reasoning.

Per-step reward summary:
  profile_column()      +0.05 first call, +0.02 repeat, -0.01 after 3x same col
  inspect_sample()      +0.03 first call, -0.01 repeated
  write_transform()     0.0  (neutral — code not yet tested)
  execute_transform()   +0.10 success, -0.08 syntax error, -0.05 runtime error
  validate()            +0.12 per NEW passing check (0 if already passing)
                        -0.02 per check that newly FAILED (regression)
  fix_transform()       +0.06 if error_msg references actual error seen
  load_to_target()      +0.05 if schema matches target, -0.05 if not
  submit()              triggers full grader (see grade_episode below)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# Per-step reward constants
# ─────────────────────────────────────────────────────────────

R_PROFILE_FIRST      =  0.05
R_PROFILE_REPEAT     =  0.02
R_PROFILE_OVERUSE    = -0.01   # >3 calls on same column

R_INSPECT_FIRST      =  0.03
R_INSPECT_REPEAT     = -0.01

R_WRITE_TRANSFORM    =  0.00   # neutral — nothing executed yet

R_EXECUTE_SUCCESS    =  0.10
R_EXECUTE_SYNTAX_ERR = -0.08
R_EXECUTE_RUNTIME_ERR = -0.05

R_VALIDATE_NEW_PASS  =  0.12   # per check that newly passes this call
R_VALIDATE_REGRESS   = -0.02   # per check that newly fails (regression)

R_FIX_GROUNDED       =  0.06   # fix references actual error
R_FIX_UNGROUNDED     =  0.01   # fix ignores error message

R_LOAD_SCHEMA_MATCH  =  0.05
R_LOAD_SCHEMA_FAIL   = -0.05

# Penalty for submitting without any validation
R_SUBMIT_NO_VALIDATE = -0.20

# Step budget efficiency (applied at submit)
EFFICIENCY_BONUS_MAX =  0.10   # extra reward for using few steps


# ─────────────────────────────────────────────────────────────
# Quality check functions (all return float 0.0–1.0)
# ─────────────────────────────────────────────────────────────

def check_null_rate(df: pd.DataFrame, required_non_null: List[str]) -> Tuple[float, str]:
    """Score: 1.0 if no nulls in required columns, else fraction passing."""
    if not required_non_null:
        return 1.0, "No non-null constraints"
    results = []
    details = []
    for col in required_non_null:
        if col not in df.columns:
            results.append(0.0)
            details.append(f"{col}: missing column")
            continue
        null_count = int(df[col].isna().sum())
        score = 1.0 if null_count == 0 else max(0.0, 1.0 - null_count / len(df))
        results.append(score)
        if null_count > 0:
            details.append(f"{col}: {null_count} nulls")
    overall = sum(results) / len(results)
    detail = "; ".join(details) if details else "All non-null constraints satisfied"
    return round(overall, 4), detail


def check_type_compliance(df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[float, str]:
    """Score: fraction of columns with correct dtype."""
    results = []
    details = []
    for col, spec in schema.items():
        if col not in df.columns:
            results.append(0.0)
            details.append(f"{col}: missing")
            continue
        expected = spec.get("dtype", "object")
        actual = str(df[col].dtype)
        # Flexible matching: datetime variants, int variants, etc.
        ok = _dtype_matches(actual, expected)
        results.append(1.0 if ok else 0.0)
        if not ok:
            details.append(f"{col}: got {actual}, expected {expected}")
    overall = sum(results) / max(1, len(results))
    detail = "; ".join(details) if details else "All types correct"
    return round(overall, 4), detail


def check_range(df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[float, str]:
    """Score: fraction of rows passing range constraints."""
    total_checks = 0
    passing_checks = 0
    details = []
    for col, spec in schema.items():
        if col not in df.columns:
            continue
        if "min" in spec:
            total_checks += len(df)
            passing = int((df[col] >= spec["min"]).sum())
            passing_checks += passing
            failing = len(df) - passing
            if failing > 0:
                details.append(f"{col} < {spec['min']}: {failing} rows")
        if "max" in spec:
            total_checks += len(df)
            passing = int((df[col] <= spec["max"]).sum())
            passing_checks += passing
            failing = len(df) - passing
            if failing > 0:
                details.append(f"{col} > {spec['max']}: {failing} rows")
    if total_checks == 0:
        return 1.0, "No range constraints"
    score = passing_checks / total_checks
    detail = "; ".join(details) if details else "All range constraints satisfied"
    return round(score, 4), detail


def check_uniqueness(df: pd.DataFrame, pk_columns: List[str]) -> Tuple[float, str]:
    """Score: 1.0 if no duplicate PKs, else fraction of unique rows."""
    if not pk_columns:
        return 1.0, "No PK constraint"
    missing = [c for c in pk_columns if c not in df.columns]
    if missing:
        return 0.0, f"PK columns missing: {missing}"
    dup_count = int(df.duplicated(subset=pk_columns).sum())
    score = 1.0 if dup_count == 0 else max(0.0, 1.0 - dup_count / len(df))
    detail = f"{dup_count} duplicate rows" if dup_count > 0 else "No duplicates"
    return round(score, 4), detail


def check_value_set(df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[float, str]:
    """Score: fraction of rows where categorical columns have valid values."""
    total = 0
    passing = 0
    details = []
    for col, spec in schema.items():
        if "values" not in spec or col not in df.columns:
            continue
        valid = set(spec["values"])
        col_total = len(df)
        col_pass = int(df[col].isin(valid).sum())
        total += col_total
        passing += col_pass
        failing = col_total - col_pass
        if failing > 0:
            bad_vals = df[~df[col].isin(valid)][col].value_counts().head(3).to_dict()
            details.append(f"{col}: {failing} invalid ({bad_vals})")
    if total == 0:
        return 1.0, "No value-set constraints"
    score = passing / total
    detail = "; ".join(details) if details else "All value-set constraints satisfied"
    return round(score, 4), detail


def check_schema_columns(df: pd.DataFrame, target_schema: Dict[str, Any]) -> Tuple[float, str]:
    """Score: fraction of required columns present with correct names."""
    required = set(target_schema.keys())
    present = set(df.columns)
    missing = required - present
    extra = present - required
    score = len(required & present) / len(required)
    parts = []
    if missing:
        parts.append(f"Missing: {sorted(missing)}")
    if extra:
        parts.append(f"Extra: {sorted(extra)}")
    detail = "; ".join(parts) if parts else "Schema matches"
    return round(score, 4), detail


def check_completeness(df_out: pd.DataFrame, gold_row_count: int) -> Tuple[float, str]:
    """
    Score: how close is the output row count to the expected gold count.
    Penalises both too many rows (padding) and too few (over-dropping).
    """
    if gold_row_count == 0:
        return 1.0, "N/A"
    ratio = len(df_out) / gold_row_count
    # Perfect = 1.0, score drops off in both directions
    # Allow ±10% tolerance with full credit
    if 0.90 <= ratio <= 1.10:
        score = 1.0
    elif ratio < 0.90:
        # Penalise over-dropping (agent dropped too many rows)
        score = max(0.0, ratio / 0.90)
    else:
        # Penalise padding (agent added rows somehow)
        score = max(0.0, 2.0 - ratio)
    detail = f"Output: {len(df_out)} rows, expected ~{gold_row_count} ({ratio:.1%})"
    return round(score, 4), detail


def check_referential_integrity(
    df_fact: pd.DataFrame,
    df_dim: pd.DataFrame,
    fk_col: str,
    pk_col: str,
) -> Tuple[float, str]:
    """Score: fraction of FK values that exist in the dimension table."""
    if fk_col not in df_fact.columns or pk_col not in df_dim.columns:
        return 0.0, f"Column missing: {fk_col} or {pk_col}"
    valid_keys = set(df_dim[pk_col].dropna().unique())
    fk_vals = df_fact[fk_col].dropna()
    if len(fk_vals) == 0:
        return 0.0, "No FK values to check"
    valid_mask = fk_vals.isin(valid_keys)
    score = valid_mask.mean()
    orphaned = int((~valid_mask).sum())
    detail = f"{orphaned} orphaned FKs" if orphaned > 0 else "All FKs valid"
    return round(score, 4), detail


def check_business_rule_margin(df: pd.DataFrame) -> Tuple[float, str]:
    """Score: fraction of rows with margin_pct > 0."""
    if "margin_pct" not in df.columns:
        return 0.0, "margin_pct column missing"
    valid = (df["margin_pct"] > 0) & df["margin_pct"].notna()
    score = float(valid.mean())
    failing = int((~valid).sum())
    detail = f"{failing} rows with margin_pct <= 0" if failing > 0 else "All margins positive"
    return round(score, 4), detail


# ─────────────────────────────────────────────────────────────
# Reward engine
# ─────────────────────────────────────────────────────────────

class RewardEngine:
    """
    Computes rewards for every ETL agent action.

    Designed for GRPO training:
      - Dense per-step signals (not sparse)
      - All signals deterministic (no LLM judge per step)
      - Reward hacking prevention built in
      - One LLM judge call at episode end (submit reasoning)
    """

    def __init__(self, task_id: str, target_schema: Dict[str, Any]):
        self.task_id = task_id
        self.target_schema = target_schema

    # ── Per-step rewards ────────────────────────────────────────

    def reward_profile_column(
        self,
        col: str,
        call_count_for_col: int,
    ) -> Tuple[float, str]:
        if call_count_for_col == 1:
            return R_PROFILE_FIRST, f"First profile of '{col}' — information gained"
        elif call_count_for_col == 2:
            return R_PROFILE_REPEAT, f"Repeat profile of '{col}'"
        else:
            return R_PROFILE_OVERUSE, f"Overusing profile on '{col}' — diminishing returns"

    def reward_inspect_sample(self, call_count: int) -> Tuple[float, str]:
        if call_count == 1:
            return R_INSPECT_FIRST, "Initial data inspection"
        return R_INSPECT_REPEAT, "Repeated inspection (consider profiling instead)"

    def reward_write_transform(self) -> Tuple[float, str]:
        return R_WRITE_TRANSFORM, "Transform stored — execute to test it"

    def reward_execute_transform(
        self,
        success: bool,
        error_type: Optional[str],
        rows_in: int,
        rows_out: int,
    ) -> Tuple[float, str]:
        if success:
            drop_pct = (rows_in - rows_out) / max(1, rows_in)
            # Penalise catastrophic row drops (agent over-filtered)
            if drop_pct > 0.50:
                penalty = -0.05 * (drop_pct - 0.50) / 0.50
                return round(R_EXECUTE_SUCCESS + penalty, 4), \
                    f"Executed OK but dropped {drop_pct:.0%} of rows — check filters"
            return R_EXECUTE_SUCCESS, f"Executed OK: {rows_in}→{rows_out} rows"
        if error_type == "syntax":
            return R_EXECUTE_SYNTAX_ERR, "Syntax error — fix the code before executing"
        return R_EXECUTE_RUNTIME_ERR, f"Runtime error: {error_type}"

    def reward_validate(
        self,
        new_scores: Dict[str, float],
        prev_scores: Dict[str, float],
    ) -> Tuple[float, str]:
        """
        Reward for NEW improvements only.
        This prevents reward hacking via repeated validate() calls.
        """
        total_r = 0.0
        parts = []
        for check, score in new_scores.items():
            prev = prev_scores.get(check, 0.0)
            delta = score - prev
            if delta > 0.01:    # genuinely improved
                r = R_VALIDATE_NEW_PASS * delta
                total_r += r
                parts.append(f"{check}: {prev:.2f}→{score:.2f} (+{r:.3f})")
            elif delta < -0.01:  # regressed (transform made things worse)
                r = R_VALIDATE_REGRESS * abs(delta)
                total_r += r
                parts.append(f"{check}: REGRESSED {prev:.2f}→{score:.2f} ({r:.3f})")
        if not parts:
            total_r = -0.01  # no improvement — small nudge to try differently
            parts = ["No improvement vs last validation"]
        return round(total_r, 4), "; ".join(parts)

    def reward_fix_transform(
        self,
        new_code: str,
        error_msg: str,
    ) -> Tuple[float, str]:
        # Simple heuristic: does the new code reference key terms from the error?
        error_keywords = re.findall(r"'(\w+)'|(\w+Error)", error_msg)
        keywords = {kw for pair in error_keywords for kw in pair if kw}
        if any(kw.lower() in new_code.lower() for kw in keywords):
            return R_FIX_GROUNDED, "Fix addresses the specific error"
        return R_FIX_UNGROUNDED, "Fix written but doesn't clearly reference the error — check carefully"

    def reward_load_to_target(
        self,
        df_out: pd.DataFrame,
    ) -> Tuple[float, str]:
        schema_score, detail = check_schema_columns(df_out, self.target_schema)
        if schema_score >= 0.9:
            return R_LOAD_SCHEMA_MATCH, f"Schema matches target ({detail})"
        return R_LOAD_SCHEMA_FAIL, f"Schema mismatch — {detail}"

    def reward_submit_early_penalty(self, has_validated: bool) -> float:
        """Penalise submitting without any validation attempt."""
        return 0.0 if has_validated else R_SUBMIT_NO_VALIDATE

    # ── Episode-final grader ────────────────────────────────────

    def grade_easy_episode(
        self,
        df_out: pd.DataFrame,
        gold_row_count: int,
        faults_planted: List[str],
        steps_taken: int,
        total_steps: int,
    ) -> Dict[str, float]:
        """
        Final deterministic grader for Easy task.
        Returns component scores AND overall (0.0–1.0).
        """
        schema = self.target_schema

        null_score,       null_detail       = check_null_rate(df_out,
            [c for c, s in schema.items() if not s.get("nullable", True)])

        type_score,       type_detail       = check_type_compliance(df_out, schema)
        range_score,      range_detail      = check_range(df_out, schema)
        unique_score,     unique_detail     = check_uniqueness(df_out,
            [c for c, s in schema.items() if s.get("unique", False)])
        value_score,      value_detail      = check_value_set(df_out, schema)
        schema_score,     schema_detail     = check_schema_columns(df_out, schema)
        complete_score,   complete_detail   = check_completeness(df_out, gold_row_count)

        # Weighted final score
        raw = (
            null_score      * 0.20 +
            type_score      * 0.20 +
            range_score     * 0.15 +
            unique_score    * 0.15 +
            value_score     * 0.10 +
            schema_score    * 0.10 +
            complete_score  * 0.10
        )

        # Efficiency bonus (using fewer steps = up to +0.10)
        efficiency = max(0.0, 1.0 - steps_taken / total_steps)
        efficiency_bonus = EFFICIENCY_BONUS_MAX * efficiency

        final = min(1.0, raw + efficiency_bonus)

        return {
            "final_score":   round(final, 4),
            "null_score":    round(null_score, 4),
            "type_score":    round(type_score, 4),
            "range_score":   round(range_score, 4),
            "unique_score":  round(unique_score, 4),
            "value_score":   round(value_score, 4),
            "schema_score":  round(schema_score, 4),
            "complete_score": round(complete_score, 4),
            "efficiency_bonus": round(efficiency_bonus, 4),
            # Diagnostics
            "details": {
                "null": null_detail, "type": type_detail,
                "range": range_detail, "unique": unique_detail,
                "value": value_detail, "schema": schema_detail,
                "completeness": complete_detail,
            }
        }

    def grade_medium_episode(
        self,
        df_fact: pd.DataFrame,
        df_customers_clean: pd.DataFrame,
        df_products_clean: pd.DataFrame,
        faults_planted: List[str],
        steps_taken: int,
        total_steps: int,
    ) -> Dict[str, float]:
        """Final grader for Medium task — adds referential integrity + business rules."""
        schema = self.target_schema

        null_score,   _  = check_null_rate(df_fact,
            [c for c, s in schema.items() if not s.get("nullable", True)])
        type_score,   _  = check_type_compliance(df_fact, schema)
        range_score,  _  = check_range(df_fact, schema)
        schema_score, _  = check_schema_columns(df_fact, schema)

        ri_cust_score, ri_cust_detail = check_referential_integrity(
            df_fact, df_customers_clean, "customer_id", "customer_id"
        ) if "customer_id" in df_fact.columns else (0.0, "missing")

        margin_score, margin_detail = check_business_rule_margin(df_fact)

        region_score, region_detail = check_value_set(df_fact,
            {"region": schema.get("region", {})}) if "region" in schema else (1.0, "N/A")

        # Completeness: join should not have lost too many valid orders
        # (we expect ~5% orphaned FKs to be legitimately dropped)
        expected_valid_orders = int(len(df_fact) * 0.95) + 1  # rough floor
        complete_score = min(1.0, len(df_fact) / max(1, expected_valid_orders))

        raw = (
            null_score      * 0.15 +
            type_score      * 0.15 +
            range_score     * 0.10 +
            schema_score    * 0.10 +
            ri_cust_score   * 0.20 +
            margin_score    * 0.15 +
            region_score    * 0.10 +
            complete_score  * 0.05
        )

        efficiency = max(0.0, 1.0 - steps_taken / total_steps)
        efficiency_bonus = EFFICIENCY_BONUS_MAX * efficiency
        final = min(1.0, raw + efficiency_bonus)

        return {
            "final_score":    round(final, 4),
            "null_score":     round(null_score, 4),
            "type_score":     round(type_score, 4),
            "range_score":    round(range_score, 4),
            "schema_score":   round(schema_score, 4),
            "ri_customer":    round(ri_cust_score, 4),
            "margin_score":   round(margin_score, 4),
            "region_score":   round(region_score, 4),
            "complete_score": round(complete_score, 4),
            "efficiency_bonus": round(efficiency_bonus, 4),
        }

    def grade_hard_episode(
        self,
        df_out: pd.DataFrame,
        drift_detected_before_exec: bool,
        steps_wasted_after_drift: int,
        pre_drift_rows_reprocessed: bool,
        gold_row_count: int,
        faults_planted: List[str],
        steps_taken: int,
        total_steps: int,
    ) -> Dict[str, float]:
        """Final grader for Hard task — adds schema drift recovery metrics."""
        base = self.grade_easy_episode(
            df_out, gold_row_count, faults_planted, steps_taken, total_steps
        )

        # Schema drift specific scores
        drift_detect_score = 1.0 if drift_detected_before_exec else 0.2
        recovery_score = max(0.0, 1.0 - steps_wasted_after_drift / 5.0)
        no_reprocess_score = 0.0 if pre_drift_rows_reprocessed else 1.0

        # Hard final: base quality + drift handling
        raw_base = base["final_score"] * 0.50
        drift_component = (
            drift_detect_score * 0.20 +
            recovery_score     * 0.20 +
            no_reprocess_score * 0.10
        )
        final = min(1.0, raw_base + drift_component)

        return {
            **base,
            "final_score":          round(final, 4),
            "drift_detect_score":   round(drift_detect_score, 4),
            "recovery_score":       round(recovery_score, 4),
            "no_reprocess_score":   round(no_reprocess_score, 4),
        }


# ─────────────────────────────────────────────────────────────
# LLM Judge — reasoning quality (1 call per episode at submit)
# ─────────────────────────────────────────────────────────────

REASONING_RUBRIC = """
You are evaluating an AI agent's reasoning log from a data cleaning task.
Score 0.0–1.0 based on this rubric:

1.0 — Reasoning correctly identifies the root cause of each data quality issue,
      references specific column names and fault types, and explains the fix chosen.
0.7 — Identifies most issues but misses 1–2, or explanations are vague.
0.4 — Identifies some issues but reasoning is generic or partially incorrect.
0.1 — Reasoning is irrelevant, hallucinated, or makes no reference to actual data.
0.0 — No reasoning provided.

Agent reasoning log:
{reasoning_log}

Identified faults summary (for your reference):
{faults_planted}

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "justification": "<one sentence>"}}
"""


def score_reasoning_with_llm(
    reasoning_log: List[str],
    faults_planted: List[str],
    openai_client,
    model: str = "gpt-4o-mini",
) -> Tuple[float, str]:
    """
    ONE LLM call per episode at submit() time.
    Returns (score, justification).

    Uses gpt-4o-mini by default (cheap, fast, sufficient for rubric scoring).
    Falls back to 0.5 if API call fails — training continues uninterrupted.
    """
    import json

    prompt = REASONING_RUBRIC.format(
        reasoning_log="\n".join(f"[Step {i+1}] {r}" for i, r in enumerate(reasoning_log) if r),
        faults_planted=", ".join(faults_planted),
    )
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()
        # Parse JSON from response
        data = json.loads(text)
        return float(data["score"]), data.get("justification", "")
    except Exception as e:
        # Fallback: don't crash training on judge failure
        return 0.5, f"Judge unavailable ({e})"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _dtype_matches(actual: str, expected: str) -> bool:
    """Flexible dtype comparison."""
    expected = expected.lower()
    actual = actual.lower()
    if expected == actual:
        return True
    # int variants
    if expected in ("int64", "int32", "int") and "int" in actual:
        return True
    # float variants
    if expected in ("float64", "float32", "float", "decimal(10,2)") and "float" in actual:
        return True
    # datetime variants
    if "datetime" in expected and "datetime" in actual:
        return True
    # object/string
    if expected in ("object", "string", "str") and actual in ("object", "string"):
        return True
    return False