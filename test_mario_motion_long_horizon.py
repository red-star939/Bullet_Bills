"""Compare local one-step and accumulated open-loop Mario motion errors.

Run from mario_pj project root:
    python test_mario_motion_long_horizon.py

Requires agent_env.py, models/mario_motion_predictor.py, and map_data/object_tiles.csv.
This script never modifies the game environment or the predictor.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agent_env import create_env
from models.mario_motion_predictor import MarioMotionPredictor, MarioState


SEED = 42
MAX_STEPS = 60
REPORT_EVERY = 5
RESULTS_DIR = Path("logs/mario_motion_validation")
TILE_CSV = Path("map_data/object_tiles.csv")


@dataclass(frozen=True)
class Scenario:
    name: str
    segments: tuple[tuple[int, float, float], ...]

    def actions(self, max_steps=MAX_STEPS):
        sequence = []
        for count, horizontal, jump in self.segments:
            sequence.extend([(horizontal, jump)] * count)
        return sequence[:max_steps]


SCENARIOS = [
    Scenario("still", ((60, 0.0, 0.0),)),
    Scenario("reverse_ground", ((8, 1.0, 0.0), (12, -1.0, 0.0),
                                (12, 1.0, 0.0), (28, 0.0, 0.0))),
    Scenario("jump_then_right", ((6, 0.0, 1.0), (12, 1.0, 0.0),
                                 (12, 0.0, 0.0), (30, -1.0, 0.0))),
    Scenario("right_jump_reverse", ((4, 1.0, 0.0), (8, 1.0, 1.0),
                                    (16, -1.0, 0.0), (32, 0.0, 0.0))),
    Scenario("alternating_jump", ((10, 0.0, 1.0), (10, 0.0, 0.0),
                                  (10, 0.0, 1.0), (10, 0.0, 0.0),
                                  (20, 0.0, 1.0))),
    Scenario("mixed", ((5, -1.0, 0.0), (5, -1.0, 1.0),
                        (7, 1.0, 0.0), (8, 1.0, 1.0),
                        (10, 0.0, 0.0), (10, -1.0, 1.0),
                        (15, 1.0, 0.0))),
]


def comparison(pred: MarioState, actual):
    """Signed prediction minus actual: x/y in px, vx/vy in px/s."""
    return {
        "dx": pred.left - int(actual[0]),
        "dy": pred.top - int(actual[1]),
        "dvx": pred.vx - float(actual[4]),
        "dvy": pred.vy - float(actual[5]),
        "grounded_match": int(pred.grounded == (not bool(actual[7]))),
    }


def max_abs(rows, key):
    return max((abs(row[key]) for row in rows), default=0.0)


def run_scenario(scenario: Scenario, predictor: MarioMotionPredictor):
    env = create_env(render_mode="none")
    rows = []
    termination = "max_steps"
    try:
        obs, info = env.reset(seed=SEED)
        # An open-loop state is never corrected after reset.
        open_state = MarioState.from_observation(obs["mario"])
        previous_time = float(info["time_elapsed"])
        expected_dt = predictor.cfg.frame_skip / predictor.cfg.fps
        actions = scenario.actions()

        print(f"\n[{scenario.name}] max_steps={len(actions)} seed={SEED}")
        print("step   t(s) action   local(dx,dy)  open(dx,dy)   open(dvx,dvy)  flags")

        for step_num, action in enumerate(actions, start=1):
            # Local estimate starts from the exact previous observation.
            local_state = MarioState.from_observation(obs["mario"])
            local_next, _ = predictor.step(local_state, action)
            open_state, _ = predictor.step(open_state, action)

            obs, reward, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32)
            )
            actual = obs["mario"]
            elapsed_time = float(info["time_elapsed"])
            observed_dt = elapsed_time - previous_time
            previous_time = elapsed_time

            local_err = comparison(local_next, actual)
            open_err = comparison(open_state, actual)

            # A terminal call may run fewer than frame_skip internal frames.
            # Compare its metrics separately to avoid misdiagnosing the model.
            complete_step = abs(observed_dt - expected_dt) < 1e-5
            row = {
                "scenario": scenario.name,
                "step": step_num,
                "time_s": round(elapsed_time, 8),
                "action_x": action[0],
                "action_jump": action[1],
                "observed_dt": round(observed_dt, 8),
                "complete_step": int(complete_step),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "actual_x": int(actual[0]),
                "actual_y": int(actual[1]),
                "actual_vx": float(actual[4]),
                "actual_vy": float(actual[5]),
                "local_x": local_next.left,
                "local_y": local_next.top,
                "open_x": open_state.left,
                "open_y": open_state.top,
                "local_dx": local_err["dx"],
                "local_dy": local_err["dy"],
                "local_dvx": local_err["dvx"],
                "local_dvy": local_err["dvy"],
                "local_grounded_match": local_err["grounded_match"],
                "open_dx": open_err["dx"],
                "open_dy": open_err["dy"],
                "open_dvx": open_err["dvx"],
                "open_dvy": open_err["dvy"],
                "open_grounded_match": open_err["grounded_match"],
            }
            rows.append(row)
            flags = []
            if not complete_step:
                flags.append("partial-step")
            if abs(open_err["dx"]) > 1 or abs(open_err["dy"]) > 1:
                flags.append("drift>1px")
            if not open_err["grounded_match"]:
                flags.append("ground-mismatch")
            if step_num == 1 or step_num % REPORT_EVERY == 0 or flags or terminated or truncated:
                print(
                    f"{step_num:4d} {elapsed_time:6.3f} "
                    f"({action[0]:+2.0f},{action[1]:.0f}) "
                    f"({local_err['dx']:+4d},{local_err['dy']:+4d}) "
                    f"({open_err['dx']:+4d},{open_err['dy']:+4d}) "
                    f"({open_err['dvx']:+8.2f},{open_err['dvy']:+8.2f}) "
                    + ",".join(flags)
                )
            if terminated or truncated:
                termination = info.get("game_state", "terminated" if terminated else "truncated")
                break
    finally:
        env.close()

    # No claim of accuracy on an incomplete last environment step.
    complete = [r for r in rows if r["complete_step"]]
    summary = {
        "scenario": scenario.name,
        "steps_observed": len(rows),
        "complete_steps": len(complete),
        "last_time_s": rows[-1]["time_s"] if rows else 0,
        "end_reason": termination,
        "max_local_dx": max_abs(complete, "local_dx"),
        "max_local_dy": max_abs(complete, "local_dy"),
        "max_open_dx": max_abs(complete, "open_dx"),
        "max_open_dy": max_abs(complete, "open_dy"),
        "max_local_dvx": max_abs(complete, "local_dvx"),
        "max_local_dvy": max_abs(complete, "local_dvy"),
        "max_open_dvx": max_abs(complete, "open_dvx"),
        "max_open_dvy": max_abs(complete, "open_dvy"),
        "first_open_drift_gt_1px_step": next(
            (r["step"] for r in complete if abs(r["open_dx"]) > 1 or abs(r["open_dy"]) > 1),
            "",
        ),
        "local_ground_mismatch_steps": sum(not r["local_grounded_match"] for r in complete),
        "open_ground_mismatch_steps": sum(not r["open_grounded_match"] for r in complete),
    }
    return rows, summary


def save_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not TILE_CSV.is_file():
        raise FileNotFoundError(f"Missing collision map input: {TILE_CSV}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictor = MarioMotionPredictor.from_object_tiles_csv(TILE_CSV)
    all_rows = []
    summaries = []
    for scenario in SCENARIOS:
        rows, summary = run_scenario(scenario, predictor)
        all_rows.extend(rows)
        summaries.append(summary)

    save_csv(RESULTS_DIR / "motion_trace.csv", all_rows)
    save_csv(RESULTS_DIR / "motion_summary.csv", summaries)
    print("\n=== VALIDATION SUMMARY (complete environment steps only) ===")
    for s in summaries:
        print(
            f"{s['scenario']:20s} "
            f"steps={s['steps_observed']:2d} "
            f"local_max=({s['max_local_dx']:.0f},{s['max_local_dy']:.0f})px "
            f"open_max=({s['max_open_dx']:.0f},{s['max_open_dy']:.0f})px "
            f"first_drift={s['first_open_drift_gt_1px_step'] or '-'} "
            f"end={s['end_reason']}"
        )
    print(f"\nSaved: {RESULTS_DIR / 'motion_trace.csv'}")
    print(f"Saved: {RESULTS_DIR / 'motion_summary.csv'}")
    print("Note: This is a deterministic model comparison, not a survival benchmark.")


if __name__ == "__main__":
    main()
