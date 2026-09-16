import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score_transcript(transcript_path: Path, scenario_path: Path) -> dict:
    transcript = json.loads(transcript_path.read_text())
    scenario = json.loads(scenario_path.read_text())

    model = get_model()
    # Use the full rationale as the TA anchor — the short answer label has too little
    # semantic content to discriminate between responses meaningfully.
    optimal_emb = model.encode(scenario["task_optimal_rationale"])

    is_experimental = transcript["condition"] == "experimental"
    regina_key = "regina" if is_experimental else "agent_a"

    scores = []

    for round_data in transcript["rounds"]:
        round_num = round_data["round"]
        responses = round_data["responses"]

        regina_emb = model.encode(responses[regina_key]) if regina_key in responses else None

        for agent, content in responses.items():
            agent_emb = model.encode(content)
            ta = cosine_sim(agent_emb, optimal_emb)
            # Skip RA for Regina — cosine_sim(x, x) = 1.0 by definition, not informative.
            ra = cosine_sim(agent_emb, regina_emb) if (regina_emb is not None and agent != regina_key) else None

            scores.append(
                {
                    "round": round_num,
                    "agent": agent,
                    "ta": round(ta, 4),
                    "ra": round(ra, 4) if ra is not None else None,
                }
            )

    result = {
        "run_id": transcript["run_id"],
        "scenario_id": transcript["scenario_id"],
        "condition": transcript["condition"],
        "run": transcript["run"],
        "scores": scores,
    }

    out_path = transcript_path.with_suffix(".scores.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Scores saved to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Score a transcript for TA and RA")
    parser.add_argument("transcript", help="Path to transcript JSON")
    parser.add_argument("scenario", help="Path to scenario JSON")
    args = parser.parse_args()
    score_transcript(Path(args.transcript), Path(args.scenario))


if __name__ == "__main__":
    main()
