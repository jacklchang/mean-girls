import argparse
import json
import time
from pathlib import Path

from openai import OpenAI, APIError, RateLimitError
from utils import (
    BASELINE_ORDER,
    EXPERIMENTAL_ORDER,
    PERSONAS_DIR,
    build_user_message,
    load_persona,
    load_scenario,
)

client = OpenAI()
RESULTS_DIR = Path(__file__).parent.parent / "results"


def call_with_retry(messages: list[dict], model: str, max_retries: int = 6) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            usage = response.usage
            cost = (usage.prompt_tokens / 1_000_000 * 2.50) + (usage.completion_tokens / 1_000_000 * 10.00)
            print(f"    tokens: {usage.prompt_tokens}in / {usage.completion_tokens}out — ${cost:.4f}")
            time.sleep(65)  # 20K tokens/call: 65s sleep + ~10s API ≈ 75s cycle, keeps calls outside each other's 60s TPM window
            return response.choices[0].message.content.strip(), cost
        except RateLimitError:
            # TPM limits reset per minute — wait at least 60s, longer on repeat hits
            wait = 60 * (attempt + 1)
            print(f"  Rate limited (TPM). Retrying in {wait}s...")
            time.sleep(wait)
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            print(f"  API error: {e}. Retrying...")
            time.sleep(5)
    raise RuntimeError("Max retries exceeded")  # pragma: no cover


def run_experiment(
    scenario_path: Path,
    condition: str,
    run_number: int,
    n_rounds: int,
    personas_dir: Path,
    output_dir: Path,
    skip_scoring: bool = False,
    start_round: int = 1,
) -> dict:
    total_cost = 0.0
    run_start = time.time()
    scenario = load_scenario(scenario_path)
    agent_names = EXPERIMENTAL_ORDER if condition == "experimental" else BASELINE_ORDER
    personas = {name: load_persona(name, personas_dir) for name in agent_names}

    seed_position = scenario.get("seed_position")
    run_id = f"{scenario['id']}_{condition}_run{run_number:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_id}.json"

    # Resume from checkpoint if start_round > 1
    if start_round > 1 and out_path.exists():
        transcript = json.loads(out_path.read_text())
        history: list[dict] = []
        if condition == "experimental" and seed_position:
            history.append({"agent": "regina", "round": 0, "content": seed_position})
        for r in transcript["rounds"]:
            for agent_name, content in r["responses"].items():
                history.append({"agent": agent_name, "round": r["round"], "content": content})
        print(f"  [resume] Loaded checkpoint with {len(transcript['rounds'])} rounds, resuming from round {start_round}")
    else:
        history = []
        if condition == "experimental" and seed_position:
            history.append({"agent": "regina", "round": 0, "content": seed_position})
            print(f"  [seed] Regina: {seed_position[:120]}...")
        transcript = {
            "run_id": run_id,
            "scenario_id": scenario["id"],
            "condition": condition,
            "run": run_number,
            "n_rounds": n_rounds,
            "seed_position": seed_position if condition == "experimental" else None,
            "rounds": [],
        }

    print(f"\n{'='*60}")
    print(f"  Scenario : {scenario['id']} — {scenario['title']}")
    print(f"  Condition: {condition}  |  Run: {run_number}  |  Rounds: {n_rounds}")
    print(f"  Agents   : {', '.join(agent_names)}")
    if start_round > 1:
        print(f"  Resuming from round {start_round}")
    print(f"{'='*60}")

    for round_num in range(start_round, n_rounds + 1):
        round_start = time.time()
        elapsed_total = round_start - run_start
        if round_num > 1:
            avg_round = elapsed_total / (round_num - 1)
            eta = avg_round * (n_rounds - round_num + 1)
            print(f"\n--- Round {round_num}/{n_rounds}  (elapsed {elapsed_total/60:.1f}m, ETA ~{eta/60:.1f}m, cost so far ${total_cost:.4f}) ---")
        else:
            print(f"\n--- Round {round_num}/{n_rounds} ---")

        round_history = history.copy()
        round_responses = []

        for agent_name in agent_names:
            agent_start = time.time()
            user_msg = build_user_message(scenario["prompt"], round_history, agent_name)

            content, cost = call_with_retry(
                messages=[
                    {"role": "system", "content": personas[agent_name]},
                    {"role": "user", "content": user_msg},
                ],
                model="gpt-5.6-luna",
            )
            total_cost += cost
            agent_elapsed = time.time() - agent_start
            turn = {"agent": agent_name, "round": round_num, "content": content}

            round_history.append(turn)
            round_responses.append(turn)
            print(f"  [{agent_elapsed:.1f}s] {agent_name}: {content[:160]}...")

        history.extend(round_responses)
        transcript["rounds"].append(
            {
                "round": round_num,
                "responses": {t["agent"]: t["content"] for t in round_responses},
            }
        )
        round_elapsed = time.time() - round_start
        print(f"  -- round done in {round_elapsed:.1f}s, cumulative cost ${total_cost:.4f}")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{transcript['run_id']}.json"
        out_path.write_text(json.dumps(transcript, indent=2))
        print(f"  [checkpoint] saved after round {round_num}")
        # Pace rounds to avoid hitting TPM limits (~82K tokens/round at 20K context).
        # If the round finished in under 90s, sleep the remainder so each round takes
        # at least 90s and the per-minute token rate stays under the limit.
        if round_num < n_rounds:
            min_round_time = 90
            pause = max(0, min_round_time - round_elapsed)
            if pause > 0:
                print(f"  [pacing] sleeping {pause:.0f}s to respect TPM limit...")
                time.sleep(pause)

    total_elapsed = time.time() - run_start
    out_path.write_text(json.dumps(transcript, indent=2))
    print(f"\n{'='*60}")
    print(f"  Saved  : {out_path}")
    print(f"  Cost   : ${total_cost:.4f}")
    print(f"  Time   : {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    if not skip_scoring:
        try:
            from score import score_transcript
            print("Scoring...")
            score_transcript(out_path, scenario_path)
            print("Scores saved.")
        except ImportError:
            print("(sentence-transformers not available — run score.py locally to score)")

    return transcript


def main():
    parser = argparse.ArgumentParser(description="Run a Mean Girls multi-agent experiment")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON")
    parser.add_argument("--condition", choices=["experimental", "baseline"], default="experimental")
    parser.add_argument("--run", type=int, default=1, help="Run number (for labeling)")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--personas-dir", type=Path, default=PERSONAS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--skip-scoring", action="store_true", help="Skip auto-scoring (score locally after run)")
    parser.add_argument("--start-round", type=int, default=1, help="Resume from this round (requires checkpoint file)")
    args = parser.parse_args()

    run_experiment(
        scenario_path=Path(args.scenario),
        condition=args.condition,
        run_number=args.run,
        n_rounds=args.rounds,
        personas_dir=args.personas_dir,
        output_dir=args.output_dir,
        skip_scoring=args.skip_scoring,
        start_round=args.start_round,
    )


if __name__ == "__main__":
    main()
