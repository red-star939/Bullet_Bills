"""models2 inheriting runner: v10 + Spawn Shield + Altitude v1/v2/v3.

Risk tracking is observation-only. FailureAnalyzer is diagnostic-only.
This does not modify PPO checkpoints or original models modules.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import gymnasium as gym
import kymnasium
import numpy as np
import torch
import agent_ppo_v10 as v10
from models.v10_spawn_guidance import V10SpawnGuidance
from models.adaptive_altitude_controller import AdaptiveAltitudeController
from models2.tracking import RiskObserverV2
from models2.altitude_v2 import AdaptiveAltitudeV2, V2Config
from models2.altitude_v3 import AdaptiveAltitudeV3, V3Config
from models.failure_analyzer import FailureConfig
from models.failure_analyzer_v2 import FailureAnalyzerV2, EngineCollisionAudit


class DiagnosticEvaluator:
    rankings = ()
    horizon_steps = 0
    strategy = 'v10_hybrid_diagnostic_no_action_rankings'


def safe_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(x) for x in value]
    return value


def run(args):
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)
    torch.set_num_threads(args.torch_threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agent = v10.BulletBillAgent.load(args.model)
    agent.deterministic = not args.stochastic
    guide = V10SpawnGuidance(map_path=args.map, caution_px=args.caution_px,
                             critical_px=args.critical_px,
                             safety_horizon_steps=args.safety_horizon_steps,
                             spawn_horizon_steps=args.spawn_horizon_steps)
    altitude = AdaptiveAltitudeController(guide.guide.simulator)
    risk = RiskObserverV2(guide.guide.simulator)
    altitude_v2 = AdaptiveAltitudeV2(guide.guide.simulator, V2Config(
        horizon_steps=args.v2_horizon, prefix_steps=args.v2_prefix,
        replan_every=args.v2_replan_every, lower_min=args.v2_lower_min,
        lower_advantage=args.v2_lower_advantage, max_candidates=args.v2_max_candidates))
    altitude_v3 = AdaptiveAltitudeV3(guide.guide.simulator, v3_config=V3Config(
        mode='shadow' if args.altitude == 'v3-shadow' else 'active',
        imminent_steps=args.v3_imminent_steps, audit_horizon_steps=args.v3_audit_horizon),
        v2_config=V2Config(horizon_steps=args.v2_horizon, prefix_steps=args.v2_prefix,
                           replan_every=args.v2_replan_every, lower_min=args.v2_lower_min,
                           lower_advantage=args.v2_lower_advantage,
                           max_candidates=args.v2_max_candidates))
    failure = FailureAnalyzerV2(FailureConfig(history_seconds=args.history_seconds))
    env = gym.make(id=v10.ENV_ID, render_mode=args.render, bgm=False)
    clock = None
    if args.render == 'human':
        import pygame
        clock = pygame.time.Clock()
    summaries = []
    print(f'MODEL {args.model} | altitude={args.altitude} | risk=observe_only | '
          f'numba={v10.HAS_NUMBA} | render={args.render} | max_steps={args.steps}', flush=True)
    if not v10.HAS_NUMBA:
        print('WARNING numba missing: v10 action evaluation can be slow', flush=True)
    quit_requested = False
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, info = env.reset(seed=seed)
            agent.fe.reset()
            altitude.reset()
            altitude_v2.reset()
            altitude_v3.reset()
            shadow_rows = []
            risk.reset()
            failure.reset(seed, ep)
            previous_action = None
            step_count = 0
            overrides = known_overrides = spawn_overrides = altitude_overrides = risk_overrides = 0
            wall_steps = upper_steps = lower_steps = 0
            saw_bullets = 0
            new_bullets = 0
            highest_bullet_count = 0
            max_lane_load = 0
            last_risk = {}
            terminated = truncated = False
            start = time.perf_counter()
            core = env.unwrapped
            mario = getattr(core, '_mario', None)
            audit = EngineCollisionAudit(mario) if args.engine_audit and mario is not None else None
            if args.engine_audit and audit is None:
                print('WARNING engine collision hook not found; failure evidence will be observational', flush=True)
            eval_stub = DiagnosticEvaluator()
            report = None
            with (audit if audit is not None else nullcontext()):
                for step in range(args.steps):
                    if args.render == 'human':
                        import pygame
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                                quit_requested = True
                    if quit_requested:
                        break
                    before = time.perf_counter()
                    last_risk = risk.observe(obs, info.get('time_elapsed', step * v10.DT))
                    raw = agent.act(obs, info)
                    guided, meta = guide.decide(obs, raw, previous_action=previous_action, mode='spawn_shield')
                    if args.altitude == 'v1':
                        guided, ameta = altitude.decide(obs, guided)
                        ameta = {'altitude_overridden': ameta['altitude_overridden'],
                                 'altitude_reason': ameta.get('altitude_reason', '')}
                    elif args.altitude == 'v2':
                        guided, v2meta = altitude_v2.decide(obs, guided)
                        ameta = {'altitude_overridden': v2meta['altitude_v2_overridden'],
                                 'altitude_reason': v2meta['altitude_v2_reason']}
                    elif args.altitude in ('v3-shadow', 'v3-active'):
                        guided, v3meta = altitude_v3.decide(obs, guided)
                        ameta = {'altitude_overridden': v3meta['altitude_overridden'],
                                 'altitude_reason': v3meta['v3_reason']}
                        if v3meta['v2_proposed'] or v3meta['v3_overridden'] or args.shadow_all_steps:
                            shadow_rows.append({
                                'episode': ep+1, 'seed': seed, 'step': step,
                                'time_s': round(float(info.get('time_elapsed', step*v10.DT)), 6),
                                'mario_y': round(float((obs['mario'][1]+obs['mario'][3])/2), 2),
                                'visible_bullets': last_risk['visible_bullets'],
                                'min_ttc_s': last_risk['min_ttc_s'],
                                'v1_action': json.dumps(v3meta['v1_action']),
                                'v2_proposal': json.dumps(v3meta['v2_action']),
                                'v1_reason': v3meta['v1_reason'],
                                'v2_reason': v3meta['v2_reason'],
                                'v3_reason': v3meta['v3_reason'],
                                'v3_overridden': int(v3meta['v3_overridden']),
                                'plans_examined': v3meta['v2_plans_examined'],
                                'route_gain_px': v3meta['v2_route_gain_px'],
                                'active_audit': json.dumps(v3meta['v3_active_audit']),
                            })
                    else:
                        ameta = {'altitude_overridden': False, 'altitude_reason': 'off'}
                    # The proven-harmful Risk Guard is intentionally never applied here.
                    action = guided
                    rmeta = {'risk_guard_overridden': False}
                    previous_action = tuple(float(x) for x in action)
                    overrides += int(not np.array_equal(np.asarray(raw, np.float32), action))
                    known_overrides += int(meta['known_world_overridden'])
                    spawn_overrides += int(meta['spawn_overridden'])
                    altitude_overrides += int(ameta['altitude_overridden'])
                    risk_overrides += int(rmeta['risk_guard_overridden'])
                    if 'wall_clearance_px' in meta:
                        wall_steps += int(meta['wall_clearance_px'] < args.caution_px)
                    cy = float(obs['mario'][1] + obs['mario'][3]) / 2.0
                    upper_steps += int(cy < 432)
                    lower_steps += int(cy >= 432)
                    saw_bullets += last_risk['visible_bullets']
                    new_bullets += last_risk['new_bullets']
                    highest_bullet_count = max(highest_bullet_count, last_risk['visible_bullets'])
                    max_lane_load = max(max_lane_load, last_risk['max_lane_load'])
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    contacts = audit.take() if audit is not None else None
                    report = failure.record(step=step, observation=obs, action=action,
                                            evaluation=eval_stub, next_observation=next_obs,
                                            info=info, terminated=terminated, truncated=truncated,
                                            policy_compute_s=time.perf_counter() - before,
                                            engine_contacts=contacts)
                    obs = next_obs
                    step_count += 1
                    if args.profile_every and step_count % args.profile_every == 0:
                        print(f'PERF ep={ep+1} step={step_count} time={info.get("time_elapsed", 0):.1f}s '
                              f'bullets={last_risk["visible_bullets"]} new={last_risk["new_bullets"]} '
                              f'upper={last_risk["upper_bullets"]} lower={last_risk["lower_bullets"]} '
                              f'ttc={last_risk["min_ttc_s"]} '
                              f'known={known_overrides} spawn={spawn_overrides} '
                              f'altitude={altitude_overrides} risk_guard={risk_overrides}', flush=True)
                    if clock is not None and args.fps > 0:
                        clock.tick(args.fps)
                    if terminated or truncated:
                        break
                if step_count > 0 and report is None:
                    report = failure.report()
            elapsed = float(info.get('time_elapsed', step_count * v10.DT))
            status = ('user_quit' if quit_requested else 'terminated' if terminated else
                      'truncated' if truncated else 'step_limit')
            death = bool(report['is_death']) if report is not None else False
            row = {
                'episode': ep+1, 'seed': seed, 'survival_s': elapsed, 'steps': step_count,
                'status': status, 'is_death': death,
                'overrides': overrides, 'known_overrides': known_overrides,
                'spawn_overrides': spawn_overrides, 'altitude_overrides': altitude_overrides,
                'risk_overrides': risk_overrides, 'wall_steps': wall_steps,
                'altitude_version': args.altitude,
                'v2_danger_rescues': altitude_v2.danger_rescues,
                'v2_crowd_escapes': altitude_v2.crowd_escapes,
                'v3_proposals': altitude_v3.shadow.differences,
                'v3_overrides': altitude_v3.overrides,
                'v3_rejections': altitude_v3.rejected,
                'upper_fraction': upper_steps / max(1, step_count),
                'lower_fraction': lower_steps / max(1, step_count),
                'mean_visible_bullets': saw_bullets / max(1, step_count),
                'new_bullet_tracks': new_bullets,
                'max_visible_bullets': highest_bullet_count,
                'max_lane_load': max_lane_load,
                'primary_failure_label': report['primary_label'] if report else 'none',
                'engine_evidence_status': report['engine_evidence_status'] if report else 'none',
            }
            summaries.append(row)
            if args.altitude in ('v3-shadow', 'v3-active'):
                shadow_path = out / f'shadow_ep_{ep+1:03d}_seed_{seed}.csv'
                if shadow_rows:
                    with shadow_path.open('w', newline='', encoding='utf-8') as sf:
                        sw = csv.DictWriter(sf, fieldnames=list(shadow_rows[0]))
                        sw.writeheader(); sw.writerows(shadow_rows)
                else:
                    shadow_path.write_text('no_proposals\n', encoding='utf-8')
            if report is not None:
                (out / f'failure_ep_{ep+1:03d}_seed_{seed}.json').write_text(
                    json.dumps(safe_json(report), ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'EPISODE {ep+1} seed={seed} survival={elapsed:.3f}s steps={step_count} '
                  f'status={status} failure={row["primary_failure_label"]} '
                  f'known={known_overrides} spawn={spawn_overrides} altitude={altitude_overrides} '
                  f'risk_guard={risk_overrides} upper_fraction={row["upper_fraction"]:.3f} '
                  f'mean_bullets={row["mean_visible_bullets"]:.2f} '
                  f'wall_elapsed={time.perf_counter()-start:.1f}s', flush=True)
            with (out / 'episodes.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader(); w.writerows(summaries)
            if quit_requested:
                break
    finally:
        env.close()
        if args.render == 'human':
            import pygame
            pygame.quit()
    result = {
        'model': args.model, 'risk_mode': 'observe', 'altitude': args.altitude, 'seed': args.seed,
        'episodes': len(summaries), 'cap_seconds': args.steps * v10.DT,
        'mean_observed_survival_s': sum(x['survival_s'] for x in summaries) / max(1, len(summaries)),
        'max_observed_survival_s': max((x['survival_s'] for x in summaries), default=0),
        'reached_300s': sum(x['survival_s'] >= 300 for x in summaries),
        'reached_600s': sum(x['survival_s'] >= 600 for x in summaries),
        'step_limit_count': sum(x['status'] == 'step_limit' for x in summaries),
        'failure_counts': dict(Counter(x['primary_failure_label'] for x in summaries)),
        'note': 'Observed survival is right-censored at step limit; engine audit is diagnostic only.',
    }
    (out / 'summary.json').write_text(json.dumps(safe_json(result), indent=2, ensure_ascii=False), encoding='utf-8')
    print('SUMMARY', json.dumps(safe_json(result), ensure_ascii=False), flush=True)
    print('OUTPUT', out.resolve(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='models/bullet_bill_ppo_v10_best.pt')
    p.add_argument('--altitude', choices=['v1', 'v2', 'v3-shadow', 'v3-active', 'off'], default='v2')
    p.add_argument('--v3-imminent-steps', type=int, default=2)
    p.add_argument('--v3-audit-horizon', type=int, default=6)
    p.add_argument('--shadow-all-steps', action='store_true')
    p.add_argument('--v2-horizon', type=int, default=18)
    p.add_argument('--v2-prefix', type=int, default=4)
    p.add_argument('--v2-replan-every', type=int, default=5)
    p.add_argument('--v2-lower-min', type=int, default=8)
    p.add_argument('--v2-lower-advantage', type=int, default=5)
    p.add_argument('--v2-max-candidates', type=int, default=18)
    p.add_argument('--render', choices=['human', 'none'], default='human')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--steps', type=int, default=9000)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--fps', type=float, default=0)
    p.add_argument('--profile-every', type=int, default=100)
    p.add_argument('--torch-threads', type=int, default=1)
    p.add_argument('--map', default='map_data/object_tiles.csv')
    p.add_argument('--caution-px', type=int, default=180)
    p.add_argument('--critical-px', type=int, default=96)
    p.add_argument('--safety-horizon-steps', type=int, default=12)
    p.add_argument('--spawn-horizon-steps', type=int, default=12)
    p.add_argument('--history-seconds', type=float, default=4)
    p.add_argument('--engine-audit', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--stochastic', action='store_true')
    p.add_argument('--out', default='logs/models2_v2')
    args = p.parse_args()
    if args.episodes < 1 or args.steps < 1 or args.torch_threads < 1 or args.fps < 0 or args.v2_prefix < 1 or args.v2_horizon <= args.v2_prefix or args.v2_max_candidates < 1:
        p.error('Invalid numeric parameter')
    run(args)


if __name__ == '__main__':
    main()
