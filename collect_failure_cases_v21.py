"""Collect failure v2.1 with exact engine death path + bullet identity (diagnostics only)."""
from __future__ import annotations
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
from agent_env import create_env
from models.action_evaluator import ActionEvaluator
from models.failure_analyzer import FailureConfig
from models.failure_analyzer_v21 import (FailureAnalyzerV21, EngineDeathAuditV21,
    snapshot_engine_bullets, summarize_reports_v21)
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--steps',type=int,default=900)
    p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--horizon-steps',type=int,default=8)
    p.add_argument('--history-seconds',type=float,default=2.)
    p.add_argument('--report-every',type=int,default=10)
    p.add_argument('--out',default='logs/failure_analyzer_v21')
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    forward=ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv('map_data/object_tiles.csv'))
    evaluator=ActionEvaluator(forward,horizon_steps=args.horizon_steps,robust_mode='none')
    analyzer=FailureAnalyzerV21(FailureConfig(history_seconds=args.history_seconds))
    rng=random.Random(args.seed);reports=[]
    for ep in range(args.episodes):
        seed=rng.randrange(2**32);rng.randrange(2**32)
        env=create_env(render_mode='none')
        try:
            obs,info=env.reset(seed=seed);core=env.unwrapped
            analyzer.reset(seed,ep);previous_action=None
            with EngineDeathAuditV21(core,world_height=forward.motion.cfg.world_height) as audit:
                for step in range(args.steps):
                    tic=time.perf_counter()
                    evaluation=evaluator.evaluate(obs,current_action=previous_action)
                    compute=time.perf_counter()-tic
                    action=evaluation.best_action
                    predicted=forward.simulate_observation(obs,[action])
                    before=snapshot_engine_bullets(core)
                    next_obs,_,terminated,truncated,info=env.step(np.asarray(action,dtype=np.float32))
                    after=snapshot_engine_bullets(core)
                    report=analyzer.record(step=step,observation=obs,action=action,
                        evaluation=evaluation,next_observation=next_obs,info=info,
                        terminated=terminated,truncated=truncated,
                        one_step_prediction=predicted,policy_compute_s=compute,
                        engine_contacts=audit.take(),engine_deaths=audit.take_deaths(),
                        engine_bullets_before=before,engine_bullets_after=after)
                    obs,previous_action=next_obs,action
                    if report is not None:break
                else:report=analyzer.report()
            reports.append(report)
            if report['is_death']:
                (out/f'episode_{ep:05d}_seed_{seed}.json').write_text(
                    json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        finally:env.close()
        if (ep+1)%args.report_every==0:
            summary=summarize_reports_v21(reports)
            print('PROGRESS',ep+1,'deaths',summary['deaths'],
                  'physical',summary['physical_death_cause_counts'],flush=True)
    rows=[]
    for r in reports:
        fatal=r.get('fatal_projectile') or {}
        rows.append(dict(episode_index=r['episode_index'],env_seed=r['env_seed'],
            end_time_s=r['end_time_s'],is_death=r['is_death'],
            physical_death_cause=r['physical_death_cause'],
            policy_failure_diagnostic=r['policy_failure_diagnostic'],
            engine_evidence_status=r['engine_evidence_status'],
            fatal_track_id=fatal.get('track_id'),
            fatal_engine_first_audited=(fatal.get('engine_first_audited') or {}).get('time_s'),
            identity_link_quality=fatal.get('identity_link_quality'),
            fatal_engine_present_before_terminal_action=fatal.get('engine_present_before_terminal_action'),
            fatal_visible_seconds=fatal.get('visible_seconds_before_death'),
            known_world_safe_alternatives=r['counterfactual_diagnostics']['known_world_alternatives_without_sampled_death']))
    with (out/'failure_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=summarize_reports_v21(reports);summary['configuration']=vars(args)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('FINAL SUMMARY',json.dumps(summary,ensure_ascii=False));print('OUTPUT',out)

if __name__=='__main__':main()
