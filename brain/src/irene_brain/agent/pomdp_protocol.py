"""Shared text boundaries for training trajectories and streaming execution."""

from irene_brain.agent.software_environment import EnvironmentObservation


def task_header(goal: str, target_function: str | None, target_module: str | None) -> str:
    text = f"[GOAL: {goal}]\n"
    if target_function:
        text += f"[TARGET_FUNCTION: {target_function}]\n"
    text += "[PHASE: WRITE_CODE]\n"
    if target_module:
        text += f"[TARGET_MODULE: {target_module}]\n"
    return text


def observation_transition(obs: EnvironmentObservation, target_module: str | None) -> str:
    text = obs.observation_text.strip()
    if not text.startswith("[OBSERVATION:"):
        text = f"[OBSERVATION: {text}]"
    if not obs.success:
        phase = "REPAIR_SYNTAX" if "SyntaxError" in text else "REPAIR_LOGIC"
    elif obs.action_type in ("RETRIEVE_MEMORY", "READ_FILE"):
        phase = "WRITE_CODE"
    else:
        phase = "VERIFY_AND_FINISH"
    suffix = f"[PHASE: {phase}]\n"
    if phase != "VERIFY_AND_FINISH":
        suffix += f"[TARGET_MODULE: {target_module or ''}]\n"
    return f"\n{text}\n{suffix}"
