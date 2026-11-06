"""Randomized, reproducible multi-seed validation for MarioMotionPredictor.

Run from the mario_pj project root (Windows PowerShell):
    python test_mario_motion_multiseed.py --episodes 10000
    python test_mario_motion_multiseed.py --forever

Inputs: agent_env.create_env, models.mario_motion_predictor, map_data/object_tiles.csv.
Does not modify game physics or train an agent. Seeds AND action-sequence seeds
are recorded, so failures can be reproduced exactly.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import secrets
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agent_env import create_env
from models.mario_motion_predictor import MarioMotionPredictor, MarioState


TILE_CSV = Path("map_data/object_tiles.csv")
DEFAULT_OUT = Path("logs/mario_motion_multiseed")
SEED_MAX = 2**32


@dataclass(frozen=True)
class Trial:
    episode: int
    env_seed: int
    action_seed: int
    mode: str


def decide_action(mode: str, step: int, rng: random.Random, control: dict) -> tuple[float, float]:
    """Deterministic for a given (mode, action_seed), independent of env RNG."""
    if mode == "still":
        return (0.0, 0.0)
    if mode == "constant_right":
        return (1.0, 0.0)
    if mode == "constant_left":
        return (-1.0, 0.0)
    if mode == "hold_jump":
        return (0.0, 1.0)
    if mode == "jump_right":
        return (1.0, 1.0)
    if mode == "jump_left":
        return (-1.0, 1.0)
    if mode == "alternating":
        return ((1.0 if (step // 6) % 2 == 0 else -1.0),
                (1.0 if (step // 11) % 2 == 0 else 0.0))
    if mode == "mixed":
        # Hold actions for several steps to generate realistic acceleration,
        # reversals, jump arcs and platform transitions, not white-noise jitter.
        if step >= control.get("next_change", 0):
            control["action"] = (
                rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]),
                rng.choice([0.0, 0.0, 0.0, 0.5, 1.0]),
            )
            control["next_change"] = step + rng.randint(2, 12)
        return control["action"]
    if mode == "random_each_step":
        return (rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]),
                rng.choice([0.0, 0.0, 1.0]))
    raise ValueError(f"Unknown mode: {mode}")


MODES = (
    "still", "constant_right", "constant_left", "hold_jump",
    "jump_right", "jump_left", "alternating", "mixed", "random_each_step",
)


def compare(pred: MarioState, actual) -> dict:
    actual_grounded = not bool(actual[7])
    return {
        "dx": pred.left - int(actual[0]),
        "dy": pred.top - int(actual[1]),
        "dvx": pred.vx - float(actual[4]),
        "dvy": pred.vy - float(actual[5]),
        "grounded_match": pred.grounded == actual_grounded,
    }


def run_trial(trial: Trial, predictor: MarioMotionPredictor,
              max_steps: int, threshold_px: int, velocity_tol: float) -> tuple[dict, list[dict]]:
    env = create_env(render_mode="none")
    rng = random.Random(trial.action_seed)
    control = {}
    mismatch_rows = []
    summary = {
        "episode": trial.episode,
        "env_seed": trial.env_seed,
        "action_seed": trial.action_seed,
        "mode": trial.mode,
        "steps": 0,
        "complete_steps": 0,
        "partial_steps": 0,
        "max_local_x_error_px": 0,
        "max_local_y_error_px": 0,
        "max_open_x_error_px": 0,
        "max_open_y_error_px": 0,
        "max_open_vx_error": 0.0,
        "max_open_vy_error": 0.0,
        "local_ground_mismatches": 0,
        "open_ground_mismatches": 0,
        "bad_complete_steps": 0,
        "first_bad_step": "",
        "end_reason": "max_steps",
        "last_time_s": 0.0,
    }
    try:
        obs, info = env.reset(seed=trial.env_seed)
        open_state = MarioState.from_observation(obs["mario"])
        previous_time = float(info["time_elapsed"])
        expected_dt = predictor.cfg.frame_skip / predictor.cfg.fps

        for step in range(1, max_steps + 1):
            action = decide_action(trial.mode, step - 1, rng, control)
            local_state = MarioState.from_observation(obs["mario"])
            local_next, _ = predictor.step(local_state, action)
            open_state, _ = predictor.step(open_state, action)

            obs, _reward, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32)
            )
            current_time = float(info["time_elapsed"])
            observed_dt = current_time - previous_time
            previous_time = current_time
            summary["steps"] = step
            summary["last_time_s"] = round(current_time, 8)

            # Terminal env.step may execute <frame_skip internal frames.
            # Such an endpoint is not an equivalent-time comparison.
            complete = math.isclose(observed_dt, expected_dt, abs_tol=1e-5, rel_tol=0)
            if complete:
                local_err = compare(local_next, obs["mario"])
                open_err = compare(open_state, obs["mario"])
                summary["complete_steps"] += 1
                summary["max_local_x_error_px"] = max(
                    summary["max_local_x_error_px"], abs(local_err["dx"]))
                summary["max_local_y_error_px"] = max(
                    summary["max_local_y_error_px"], abs(local_err["dy"]))
                summary["max_open_x_error_px"] = max(
                    summary["max_open_x_error_px"], abs(open_err["dx"]))
                summary["max_open_y_error_px"] = max(
                    summary["max_open_y_error_px"], abs(open_err["dy"]))
                summary["max_open_vx_error"] = max(
                    summary["max_open_vx_error"], abs(open_err["dvx"]))
                summary["max_open_vy_error"] = max(
                    summary["max_open_vy_error"], abs(open_err["dvy"]))
                summary["local_ground_mismatches"] += int(not local_err["grounded_match"])
                summary["open_ground_mismatches"] += int(not open_err["grounded_match"])
                bad = (
                    abs(open_err["dx"]) > threshold_px
                    or abs(open_err["dy"]) > threshold_px
                    or abs(open_err["dvx"]) > velocity_tol
                    or abs(open_err["dvy"]) > velocity_tol
                    or not open_err["grounded_match"]
                )
                if bad:
                    summary["bad_complete_steps"] += 1
                    if not summary["first_bad_step"]:
                        summary["first_bad_step"] = step
                    mismatch_rows.append({
                        "episode": trial.episode,
                        "env_seed": trial.env_seed,
                        "action_seed": trial.action_seed,
                        "mode": trial.mode,
                        "step": step,
                        "time_s": round(current_time, 8),
                        "action_x": action[0],
                        "action_jump": action[1],
                        "actual_x": int(obs["mario"][0]),
                        "actual_y": int(obs["mario"][1]),
                        "predicted_x": open_state.left,
                        "predicted_y": open_state.top,
                        "open_dx": open_err["dx"],
                        "open_dy": open_err["dy"],
                        "open_dvx": open_err["dvx"],
                        "open_dvy": open_err["dvy"],
                        "grounded_match": int(open_err["grounded_match"]),
                    })
            else:
                summary["partial_steps"] += 1

            if terminated or truncated:
                summary["end_reason"] = info.get(
                    "game_state", "terminated" if terminated else "truncated"
                )
                break
    finally:
        env.close()
    return summary, mismatch_rows


def write_row(writer, file_handle, row):
    writer.writerow(row)
    file_handle.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--episodes", type=int, default=10_000,
                            help="Random episodes to run (default: 10000)")
    mode_group.add_argument("--forever", action="store_true",
                            help="Run until Ctrl+C (results continuously saved)")
    parser.add_argument("--steps", type=int, default=60,
                        help="Maximum actions per episode (default: 60)")
    parser.add_argument("--threshold-px", type=int, default=1,
                        help="Allowed absolute open-loop position error (default: 1)")
    parser.add_argument("--velocity-tol", type=float, default=1e-3,
                        help="Allowed absolute vx/vy error (default: 0.001)")
    parser.add_argument("--report-every", type=int, default=100,
                        help="Print aggregated progress every N episodes")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rng-seed", type=int, default=None,
                        help="Optional RNG seed for reproducing the entire trial schedule")
    args = parser.parse_args()
    if args.episodes <= 0 or args.steps <= 0 or args.report_every <= 0:
        parser.error("episodes, steps, report-every must be positive")
    if args.threshold_px < 0 or args.velocity_tol < 0:
        parser.error("tolerances must be nonnegative")
    if not TILE_CSV.is_file():
        raise FileNotFoundError(f"Missing {TILE_CSV}; run from project root")

    args.out.mkdir(parents=True, exist_ok=True)
    predictor = MarioMotionPredictor.from_object_tiles_csv(TILE_CSV)
    schedule_rng = random.Random(args.rng_seed) if args.rng_seed is not None else random.SystemRandom()
    seen_seeds = set()

    summary_path = args.out / "episode_summary.csv"
    failures_path = args.out / "mismatch_steps.csv"
    summary_fields = [
        "episode", "env_seed", "action_seed", "mode", "steps", "complete_steps",
        "partial_steps", "max_local_x_error_px", "max_local_y_error_px",
        "max_open_x_error_px", "max_open_y_error_px",
        "max_open_vx_error", "max_open_vy_error",
        "local_ground_mismatches", "open_ground_mismatches",
        "bad_complete_steps", "first_bad_step", "end_reason", "last_time_s",
    ]
    mismatch_fields = [
        "episode", "env_seed", "action_seed", "mode", "step", "time_s",
        "action_x", "action_jump", "actual_x", "actual_y", "predicted_x",
        "predicted_y", "open_dx", "open_dy", "open_dvx", "open_dvy", "grounded_match",
    ]
    totals = {"episodes": 0, "complete_steps": 0, "bad_steps": 0,
              "bad_episodes": 0, "partial_steps": 0, "max_dx": 0, "max_dy": 0}
    print("Random Mario motion validation", flush=True)
    print(f"episodes={'unlimited' if args.forever else args.episodes}; "
          f"max_steps={args.steps}; output={args.out}; rng_seed={args.rng_seed}", flush=True)
    print("Each trial records env_seed and action_seed for exact replay.", flush=True)
    try:
        with summary_path.open("w", newline="", encoding="utf-8-sig") as sf, \
             failures_path.open("w", newline="", encoding="utf-8-sig") as ff:
            sw = csv.DictWriter(sf, fieldnames=summary_fields)
            fw = csv.DictWriter(ff, fieldnames=mismatch_fields)
            sw.writeheader()
            fw.writeheader()
            episode = 0
            while args.forever or episode < args.episodes:
                episode += 1
                env_seed = schedule_rng.randrange(SEED_MAX)
                while env_seed in seen_seeds:
                    env_seed = schedule_rng.randrange(SEED_MAX)
                seen_seeds.add(env_seed)
                action_seed = schedule_rng.randrange(SEED_MAX)
                mode = MODES[(episode - 1) % len(MODES)]
                trial = Trial(episode, env_seed, action_seed, mode)
                summary, mismatches = run_trial(
                    trial, predictor, args.steps, args.threshold_px, args.velocity_tol
                )
                write_row(sw, sf, summary)
                for mismatch in mismatches:
                    write_row(fw, ff, mismatch)
                totals["episodes"] += 1
                totals["complete_steps"] += summary["complete_steps"]
                totals["bad_steps"] += summary["bad_complete_steps"]
                totals["bad_episodes"] += bool(summary["bad_complete_steps"])
                totals["partial_steps"] += summary["partial_steps"]
                totals["max_dx"] = max(totals["max_dx"], summary["max_open_x_error_px"])
                totals["max_dy"] = max(totals["max_dy"], summary["max_open_y_error_px"])
                if episode == 1 or episode % args.report_every == 0 or (mismatches and totals["bad_episodes"] <= 5):
                    print(
                        f"episodes={episode:,} complete_steps={totals['complete_steps']:,} "
                        f"bad_episodes={totals['bad_episodes']:,} "
                        f"bad_steps={totals['bad_steps']:,} "
                        f"partial_steps={totals['partial_steps']:,} "
                        f"max_open_error=({totals['max_dx']},{totals['max_dy']})px "
                        f"last_seed={env_seed} last_mode={mode}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\nStopped by user; completed trial results are saved.", flush=True)
    print("\nFinal summary:", totals, flush=True)
    print("Saved:", summary_path, failures_path, sep="\n  ", flush=True)
    print("Note: validates motion predictions, NOT policy survival/generalization.")


if __name__ == "__main__":
    main()
