"""True multi-process (Windows spawn) on-policy PPO v10 fine-tuning with auxiliary imitation of frozen v10+Spawn+Altitude.

- Executes only the CURRENT STUDENT PPO's sampled actions in training envs.
- Teacher is a deterministic frozen original v10 policy + inference overlays.
- Teacher final action supervises jump categorical and continuous move mean.
- Original v10 checkpoint schema is retained for normal BulletBillAgent.load().
- Local 2M-step training; does not claim to embed external shield in network.
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import traceback
from collections import deque
from pathlib import Path
import json
import math
import os
import random
import time

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import gymnasium as gym
import kymnasium  # register
import numpy as np
import torch
import torch.nn.functional as F

import agent_ppo_v10 as v10
from models.v10_spawn_guidance import V10SpawnGuidance
from models.adaptive_altitude_controller import AdaptiveAltitudeController


def teacher_action(teacher, guide, altitude, state, obs, previous_action):
    with torch.no_grad():
        j, m, _ = teacher.act(torch.from_numpy(state).unsqueeze(0), deterministic=True)
    proposed = v10.to_env_action(float(m.item()), int(j.item()))
    action, _ = guide.decide(obs, proposed, previous_action=previous_action, mode='spawn_shield')
    action, meta = altitude.decide(obs, action)
    jump = int(np.argmin(np.abs(v10.JUMP_LEVELS - float(action[1]))))
    return float(action[0]), jump, action, bool(meta.get('altitude_overridden',False))


def save_checkpoint(path, model, optimizer, total_steps, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'obs_dim': v10.OBS_DIM, 'hidden': 256,
        'finetune_meta': {'algo': 'on_policy_ppo_auxiliary_v10_spawn_altitude',
                          'env_steps': total_steps, 'source': cfg['init'],
                          'note': 'Neural student; external safety controller NOT embedded in weights',
                          'train_config': cfg},
        'optimizer_state_dict': optimizer.state_dict(),
    }
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def eval_student(model, seeds, max_steps):
    env = gym.make(v10.ENV_ID, render_mode='none', bgm=False)
    fe = v10.FeatureExtractor()
    durations, censored = [], []
    try:
        for seed in seeds:
            obs, info = env.reset(seed=int(seed))
            fe.reset()
            done = False
            for step in range(max_steps):
                state, _, _ = fe(obs, info)
                with torch.no_grad():
                    j, m, _ = model.act(torch.from_numpy(state).unsqueeze(0), deterministic=True)
                obs, _, term, trunc, info = env.step(v10.to_env_action(m.item(),j.item()))
                if term or trunc:
                    done = True
                    break
            durations.append(float(info.get('time_elapsed',(step+1)*v10.DT)))
            censored.append(not done)
    finally:
        env.close()
    return float(np.mean(durations)), float(np.median(durations)), int(sum(censored))


def worker_main(connection, worker_id, args, frozen_state):
    """One independent game and frozen teacher per Windows-spawn process."""
    env = None
    try:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # Distinct but reproducible worker RNG; independent from learner RNG.
        random.seed(args.seed + 100003 * worker_id)
        np.random.seed((args.seed + 100003 * worker_id) % (2**32))
        torch.manual_seed(args.seed + 100003 * worker_id)
        env = gym.make(v10.ENV_ID, render_mode='none', bgm=False)
        fe = v10.FeatureExtractor()
        guide = V10SpawnGuidance(map_path=args.map)
        altitude = AdaptiveAltitudeController(guide.guide.simulator)
        teacher = v10.ActorCritic(obs_dim=v10.OBS_DIM, hidden=256)
        teacher.load_state_dict(frozen_state, strict=True)
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        observation, info = env.reset(seed=args.seed * 1000 + worker_id)
        state, edge, bullet = fe(observation, info)
        phi = v10.potential(edge, bullet)
        previous_action = None
        connection.send(('ready', (state, phi)))
        while True:
            command, payload = connection.recv()
            if command == 'close':
                break
            if command != 'step':
                raise ValueError(f'Unknown command: {command}')
            move, jump = payload
            # Frozen teacher sees exactly the PRE-action observation.
            tm, tj, _teach_action, altitude_override = teacher_action(
                teacher, guide, altitude, state, observation, previous_action)
            action = v10.to_env_action(move, jump)  # execute student only
            previous_action = tuple(float(x) for x in action)
            on_ground = observation['mario'][7] < 0.5
            new_obs, _, term, trunc, new_info = env.step(action)
            new_state, edge2, bullet2 = fe(new_obs, new_info)
            done = bool(term or trunc)
            dead = bool(term and new_info.get('game_state') == 'game_over')
            phi2 = 0.0 if done else v10.potential(edge2, bullet2)
            reward = v10.SURVIVE + v10.SHAPE * (v10.GAMMA * phi2 - phi)
            if on_ground and action[1] > 0:
                reward += v10.JUMP_COST
            if dead:
                reward += v10.DEATH
            if new_obs['mario'][6] > .5 and observation['mario'][6] < .5:
                reward -= v10.STAR_EARLY_PENALTY * max(
                    0., 1. - float(new_info['time_elapsed']) / v10.STAR_FREE_TIME)
            duration = float(new_info.get('time_elapsed', 0.0)) if done else None
            if done:
                new_obs, new_info = env.reset()
                fe.reset()
                altitude.reset()
                previous_action = None
                new_state, edge2, bullet2 = fe(new_obs, new_info)
                phi2 = v10.potential(edge2, bullet2)
            observation, state, phi = new_obs, new_state, phi2
            connection.send(('step', (state, float(reward), float(done),
                                      tm, tj, int(altitude_override), duration)))
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class ParallelEnvs:
    def __init__(self, args, frozen_state):
        ctx = mp.get_context('spawn')
        self.conns = []
        self.processes = []
        self.timeout_s = args.worker_timeout
        try:
            for i in range(args.envs):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=worker_main, args=(child, i, args, frozen_state),
                                      name=f'bulletbill-worker-{i:02d}')
                process.start()
                child.close()
                self.conns.append(parent)
                self.processes.append(process)
            states = []
            for conn in self.conns:
                label, payload = self._receive(conn)
                if label != 'ready':
                    raise RuntimeError(f'Unexpected worker init message: {label}')
                states.append(payload[0])
            self.initial_states = np.asarray(states, dtype=np.float32)
        except BaseException:
            self.close()
            raise

    def _receive(self, conn):
        try:
            if not conn.poll(self.timeout_s):
                raise TimeoutError(f'Worker did not respond within {self.timeout_s}s; check CPU load or worker crash')
            label, payload = conn.recv()
        except EOFError as exc:
            raise RuntimeError('A game worker exited unexpectedly. Check error output.') from exc
        if label == 'error':
            raise RuntimeError('Game worker failed:\n' + str(payload))
        return label, payload

    def step(self, moves, jumps):
        for conn, m, j in zip(self.conns, moves, jumps):
            conn.send(('step', (float(m), int(j))))
        results = []
        for conn in self.conns:
            label, payload = self._receive(conn)
            if label != 'step':
                raise RuntimeError(f'Unexpected worker message: {label}')
            results.append(payload)
        return results

    def close(self):
        for conn in self.conns:
            try:
                conn.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
        for conn in self.conns:
            conn.close()


def train(args):
    if args.steps <= 0 or args.envs <= 0 or args.rollout <= 0 or args.worker_timeout <= 0:
        raise ValueError('steps/envs/rollout must be positive')
    if args.steps % (args.envs * args.rollout) != 0:
        raise ValueError('steps must be an exact multiple of envs*rollout; use --rollout 250 and --envs 16 for 2M')
    init = Path(args.init)
    if not init.is_file():
        raise FileNotFoundError(f'Missing v10 initialization checkpoint: {init}')
    ck = torch.load(init, map_location='cpu', weights_only=True)
    if ck.get('obs_dim') != v10.OBS_DIM or ck.get('hidden') != 256:
        raise ValueError('v10 checkpoint must have obs_dim=148 and hidden=256')
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = v10.ActorCritic(obs_dim=v10.OBS_DIM, hidden=256)
    model.load_state_dict(ck['model'], strict=True)
    teacher = v10.ActorCritic(obs_dim=v10.OBS_DIM, hidden=256)
    teacher.load_state_dict(ck['model'], strict=True)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    resume_steps = 0
    if args.resume:
        resume_ck = torch.load(args.resume, map_location='cpu', weights_only=True)
        resume_steps = int(resume_ck['finetune_meta']['env_steps'])
        if resume_steps < 0 or resume_steps >= args.steps:
            raise ValueError('resume checkpoint steps must be below target --steps')
        model.load_state_dict(resume_ck['model'], strict=True)
        optimizer.load_state_dict(resume_ck['optimizer_state_dict'])
        print(f'RESUMING from {resume_steps} environment steps (environment episodes start fresh)',flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    (out/'train_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    frozen_state = {k: val.detach().cpu().clone() for k, val in teacher.state_dict().items()}
    print(f'STARTING {args.envs} independent Windows-spawn game workers...', flush=True)
    parallel = ParallelEnvs(args, frozen_state)
    s = parallel.initial_states.copy()
    ep_time = deque(maxlen=100)
    teacher_hit = deque(maxlen=1000)
    start = time.perf_counter()
    try:
        T,N = args.rollout,args.envs
        S=np.empty((T,N,v10.OBS_DIM),np.float32)
        J=np.empty((T,N),np.int64); M=np.empty((T,N),np.float32)
        LOGP=np.empty((T,N),np.float32); V=np.empty((T,N),np.float32)
        R=np.empty((T,N),np.float32); D=np.empty((T,N),np.float32)
        TJ=np.empty((T,N),np.int64); TM=np.empty((T,N),np.float32)
        n_updates=args.steps//(T*N)
        if resume_steps % (T*N) != 0:
            raise ValueError('resume step count must be aligned with envs*rollout')
        best=-float('inf')
        for update in range(1+resume_steps//(T*N),n_updates+1):
            total=(update-1)*T*N
            frac=total/args.steps
            optimizer.param_groups[0]['lr']=args.lr*(1-frac)
            imitation_coef=args.teacher_coef_start + (args.teacher_coef_end-args.teacher_coef_start)*frac
            altitude_interventions=0
            for t in range(T):
                with torch.no_grad():
                    st=torch.from_numpy(s.copy())
                    j,m,logp=model.act(st,deterministic=False)
                    val=model.value(st)
                S[t]=s; J[t]=j.numpy(); M[t]=m.numpy(); LOGP[t]=logp.numpy(); V[t]=val.numpy()
                # Send all actions before receiving any result: actual parallel work.
                results = parallel.step(M[t], J[t])
                for i, (next_state, reward, done, teacher_move, teacher_jump,
                        alt_override, duration) in enumerate(results):
                    TJ[t,i] = teacher_jump; TM[t,i] = teacher_move
                    altitude_interventions += alt_override
                    teacher_hit.append(float(teacher_jump == J[t,i] and
                                             abs(teacher_move - np.clip(M[t,i],-1,1)) < 0.2))
                    R[t,i] = reward; D[t,i] = done
                    s[i] = next_state
                    if duration is not None:
                        ep_time.append(duration)
            with torch.no_grad():
                last_v=model.value(torch.from_numpy(s)).numpy()
            adv=v10.compute_gae(R,V,D,last_v,v10.GAMMA,args.gae_lambda)
            ret=adv+V
            B=T*N
            b_s=torch.from_numpy(S.reshape(B,-1).copy())
            b_j=torch.from_numpy(J.reshape(B).copy())
            b_m=torch.from_numpy(M.reshape(B).copy())
            b_logp=torch.from_numpy(LOGP.reshape(B).copy())
            b_adv=torch.from_numpy(adv.reshape(B).copy())
            b_ret=torch.from_numpy(ret.reshape(B).copy())
            b_tj=torch.from_numpy(TJ.reshape(B).copy())
            b_tm=torch.from_numpy(TM.reshape(B).copy())
            idx=np.arange(B)
            losses=[]
            for _ in range(args.epochs):
                np.random.shuffle(idx)
                for part in np.array_split(idx,args.minibatches):
                    bidx=torch.as_tensor(part)
                    lp,ej,em,v=model.evaluate(b_s[bidx],b_j[bidx],b_m[bidx])
                    ratio=(lp-b_logp[bidx]).exp()
                    a=b_adv[bidx]; a=(a-a.mean())/(a.std(unbiased=False)+1e-8)
                    policy_loss=torch.maximum(-a*ratio,-a*ratio.clamp(1-args.clip,1+args.clip)).mean()
                    value_loss=.5*(v-b_ret[bidx]).square().mean()
                    dj,dm=model.dists(b_s[bidx])
                    teacher_loss=(F.cross_entropy(dj.logits,b_tj[bidx]) +
                                  args.move_bc_weight*F.smooth_l1_loss(dm.mean.clamp(-1,1),b_tm[bidx]))
                    loss=(policy_loss+args.vf_coef*value_loss-args.ent_jump*ej.mean()-args.ent_move*em.mean()
                          +imitation_coef*teacher_loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad)
                    optimizer.step()
                    losses.append((policy_loss.item(),value_loss.item(),teacher_loss.item()))
            steps_done=update*B
            if update%args.log_every==0 or update==1:
                speed=steps_done/max(1e-5,time.perf_counter()-start)
                print(f'STEPS {steps_done} LAST100_SURVIVAL '
                      f'{np.mean(ep_time) if ep_time else float("nan"):.3f} '
                      f'TEACHER_COEF {imitation_coef:.4f} TEACHER_AGREEMENT {np.mean(teacher_hit):.3f} '
                      f'ALTITUDE_INTERVENTIONS {altitude_interventions} '
                      f'PPO_LOSS {np.mean(losses,axis=0)[0]:.4f} BC_LOSS {np.mean(losses,axis=0)[2]:.4f} '
                      f'ENV_STEPS_PER_S {speed:.1f}',flush=True)
            if update%args.save_every==0 or update==n_updates:
                save_checkpoint(out/f'bullet_bill_ppo_v10_altitude_step_{steps_done:09d}.pt',model,optimizer,steps_done,cfg)
                save_checkpoint(out/'bullet_bill_ppo_v10_altitude_latest.pt',model,optimizer,steps_done,cfg)
                print(f'CHECKPOINT {out}/bullet_bill_ppo_v10_altitude_latest.pt',flush=True)
            if args.eval_every and (update%args.eval_every==0 or update==n_updates):
                avg,med,capped=eval_student(model,range(args.eval_seed,args.eval_seed+args.eval_episodes),args.eval_steps)
                print(f'EVAL steps={steps_done} mean={avg:.3f}s median={med:.3f}s '
                      f'censored={capped}/{args.eval_episodes} cap={args.eval_steps*v10.DT:.1f}s',flush=True)
                if avg>best:
                    best=avg
                    save_checkpoint(out/'bullet_bill_ppo_v10_altitude_best.pt',model,optimizer,steps_done,cfg)
    finally:
        parallel.close()


def main():
    mp.freeze_support()  # Windows-friendly
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init',default='models/bullet_bill_ppo_v10_best.pt')
    p.add_argument('--resume',default=None,help='Continue from this script checkpoint; --init must remain original frozen teacher')
    p.add_argument('--out',default='models/v10_altitude_guided_2m')
    p.add_argument('--map',default='map_data/object_tiles.csv')
    p.add_argument('--steps',type=int,default=2_000_000)
    p.add_argument('--envs',type=int,default=16)
    p.add_argument('--rollout',type=int,default=250,help='Per-env steps per update; 16*250=4000 total')
    p.add_argument('--seed',type=int,default=20261110)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--worker-timeout',type=float,default=240.,help='Seconds per worker response before failing')
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--minibatches',type=int,default=4)
    p.add_argument('--clip',type=float,default=.2)
    p.add_argument('--gae-lambda',type=float,default=.95)
    p.add_argument('--vf-coef',type=float,default=.5)
    p.add_argument('--ent-jump',type=float,default=.01)
    p.add_argument('--ent-move',type=float,default=.001)
    p.add_argument('--max-grad',type=float,default=.5)
    p.add_argument('--teacher-coef-start',type=float,default=.3)
    p.add_argument('--teacher-coef-end',type=float,default=.1)
    p.add_argument('--move-bc-weight',type=float,default=2.)
    p.add_argument('--log-every',type=int,default=1)
    p.add_argument('--save-every',type=int,default=5,help='5*4000=20k steps')
    p.add_argument('--eval-every',type=int,default=125,help='125*4000=500k steps; 0 disables')
    p.add_argument('--eval-episodes',type=int,default=10)
    p.add_argument('--eval-steps',type=int,default=9000)
    p.add_argument('--eval-seed',type=int,default=50000)
    args=p.parse_args()
    train(args)

if __name__=='__main__': main()
