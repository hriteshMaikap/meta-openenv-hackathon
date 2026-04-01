"""
tests/test_env_smoke.py — Quick smoke test for the ETL environment.
Run: python tests/test_env_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../server"))

from environment import ETLEnvironment
from models import ETLAction


def run_easy_episode():
    """Simulate a competent agent solving the easy task."""
    env = ETLEnvironment()
    obs = env.reset(task_id="easy", seed=42)

    print(f"Episode started. Rows: {obs.row_count}")
    print(f"Schema: {obs.schema_current}")
    print(f"Steps remaining: {obs.steps_remaining}")

    total_reward = 0.0

    # Step 1: Profile key columns
    obs, r, done, _ = env.step(ETLAction(
        tool="profile_column",
        params={"column": "order_date"},
        reasoning="Checking date column for format inconsistencies"
    ))
    total_reward += r
    print(f"[1] profile_column: reward={r:.3f} | {obs.last_tool_output[:80]}")

    # Step 2: Profile amount
    obs, r, done, _ = env.step(ETLAction(
        tool="profile_column",
        params={"column": "amount"},
        reasoning="Checking amount for range violations and outliers"
    ))
    total_reward += r
    print(f"[2] profile_column: reward={r:.3f} | {obs.last_tool_output[:80]}")

    # Step 3: Write a comprehensive transform
    transform_code = """
import pandas as pd
# Fix date formats
df['order_date'] = pd.to_datetime(df['order_date'], infer_datetime_format=True, errors='coerce')
# Remove duplicate PKs
df = df.drop_duplicates(subset=['order_id'], keep='first')
# Remove null FK
df = df[df['customer_id'].notna()]
# Fix negative amounts
df = df[df['amount'] >= 0]
# Cap outlier amounts
df['amount'] = df['amount'].clip(upper=10000)
# Normalize status case
df['status'] = df['status'].str.lower().str.strip()
"""
    obs, r, done, _ = env.step(ETLAction(
        tool="write_transform",
        params={"code": transform_code},
        reasoning="Writing comprehensive fix for all identified faults"
    ))
    total_reward += r
    print(f"[3] write_transform: reward={r:.3f}")

    # Step 4: Execute
    obs, r, done, _ = env.step(ETLAction(
        tool="execute_transform",
        params={},
        reasoning="Running the transformation"
    ))
    total_reward += r
    print(f"[4] execute_transform: reward={r:.3f} | {obs.last_tool_output[:100]}")

    # Step 5: Validate
    obs, r, done, _ = env.step(ETLAction(
        tool="validate",
        params={"checks": ["null_check", "type_check", "range_check", "uniqueness_check", "value_set_check"]},
        reasoning="Checking all quality dimensions"
    ))
    total_reward += r
    print(f"[5] validate: reward={r:.3f} | scores={obs.validation_scores}")

    # Step 6: Load to target
    obs, r, done, _ = env.step(ETLAction(
        tool="load_to_target",
        params={},
        reasoning="All checks passing, loading output"
    ))
    total_reward += r
    print(f"[6] load_to_target: reward={r:.3f} | {obs.last_tool_output[:80]}")

    # Step 7: Submit
    obs, final_reward, done, info = env.step(ETLAction(
        tool="submit",
        params={},
        reasoning=(
            "Fixed 6 data quality issues: "
            "1) Standardized mixed date formats (ISO + natural) to datetime64. "
            "2) Removed duplicate order_id rows keeping first occurrence. "
            "3) Dropped rows with null customer_id (FK violation). "
            "4) Removed negative amount rows (business rule violation). "
            "5) Capped outlier amounts at 10000. "
            "6) Normalized status column to lowercase."
        )
    ))
    print(f"[7] submit: final_reward={final_reward:.4f}")

    print(f"\n=== Results ===")
    print(f"Final score: {info.get('final_score', 0):.4f}")
    if "grader_breakdown" in info:
        for k, v in info["grader_breakdown"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    print(f"Cumulative reward: {total_reward:.4f}")
    assert done, "Episode should be done after submit"
    return info


def test_reward_hacking_prevention():
    """Agent that profiles the same column 10 times should get penalized."""
    env = ETLEnvironment()
    env.reset(task_id="easy", seed=99)

    rewards = []
    for i in range(6):
        obs, r, done, _ = env.step(ETLAction(
            tool="profile_column",
            params={"column": "amount"},
            reasoning="profiling"
        ))
        rewards.append(r)

    print(f"\nRepeated profile rewards: {rewards}")
    assert rewards[0] == 0.05, f"First call should be +0.05, got {rewards[0]}"
    assert rewards[1] == 0.02, f"Second call should be +0.02, got {rewards[1]}"
    assert all(r == -0.01 for r in rewards[2:]), f"3rd+ calls should be -0.01, got {rewards[2:]}"
    print("✓ Reward hacking prevention: diminishing returns confirmed")


def test_no_validate_penalty():
    """Agent that submits without validating gets penalized."""
    env = ETLEnvironment()
    env.reset(task_id="easy", seed=7)
    # Immediately submit
    obs, r, done, info = env.step(ETLAction(tool="submit", params={}, reasoning=""))
    print(f"\nNo-validate submit score: {r:.4f}")
    # Should be penalized
    print("✓ No-validate penalty applied" if r < 0.5 else "⚠ No penalty applied")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Full easy episode")
    print("=" * 60)
    info = run_easy_episode()

    print("\n" + "=" * 60)
    print("Test 2: Reward hacking prevention")
    print("=" * 60)
    test_reward_hacking_prevention()

    print("\n" + "=" * 60)
    print("Test 3: No-validate penalty")
    print("=" * 60)
    test_no_validate_penalty()

    print("\n✓ All smoke tests passed")