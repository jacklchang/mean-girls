import json
from pathlib import Path

EXPERIMENTAL_ORDER = ["regina", "gretchen", "karen", "cady"]
BASELINE_ORDER = ["agent_a", "agent_b", "agent_c", "agent_d"]

PERSONAS_DIR = Path(__file__).parent.parent / "personas"
SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


def load_persona(name: str, personas_dir: Path = PERSONAS_DIR) -> str:
    return (personas_dir / f"{name}.txt").read_text().strip()


def load_scenario(path: Path) -> dict:
    return json.loads(path.read_text())


def format_agent_name(name: str) -> str:
    return name.replace("_", " ").title()


def format_history(history: list[dict]) -> str:
    return "\n\n".join(f"{format_agent_name(t['agent'])}: {t['content']}" for t in history)


def build_user_message(scenario_prompt: str, history: list[dict], agent_name: str) -> str:
    msg = f"SCENARIO:\n{scenario_prompt}"
    if history:
        msg += f"\n\nGROUP DISCUSSION SO FAR:\n{format_history(history)}"
    msg += (
        f"\n\nYou are {format_agent_name(agent_name)}. "
        "Think through your reasoning before stating your position. "
        "It is now your turn to respond."
    )
    return msg
