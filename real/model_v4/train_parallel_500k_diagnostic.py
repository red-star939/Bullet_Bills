"""Model V4 independent 16-worker PPO, 500k *environment steps*.

On-policy chunks run locally in CPU worker processes; PPO runs centrally.
No PPO v10 model/weights imported. Logging tracks movement and jump behavior.
"""
from __future__ import annotations
import argparse, csv, json, multiprocessing as mp, os, queue, secrets, time, traceback
from pathlib import Path
import numpy as np
import torch
from model_v4_net import ModelV4, OBS_DIM, HIDDEN, obs_vector, env_action, make_payload

HERE = Path(__file__).resolve().parent
ENV_ID = 'kymnasium/BulletBill-Continuous-Hard-Stage-2'

def make_env():
    import gymnasium as gym
    import kymnasium  # noqa: F401
    return gym.make(id=ENV_ID, render_mode='none', bgm=False)

def atomic_save(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)

def get_mario_x(obs):
    """Best-effort player center X extraction for diagnostics; never used for reward."""
    try:
        mario = obs.get('mario') if isinstance(obs, dict) else None
        if mario is None:
            return float('nan')
        box = np.asarray(mario, dtype=np.float64).reshape(-1)
        if box.size >= 4:
            return float((box[0] + box[2]) / 2.0)
        if box.size >= 1:
            return float(box[0])
    except (TypeError, ValueError, IndexError):
        pass
    return float('nan')


def worker(index, requests, responses, master_seed, chunk_size, max_episode_steps):
    torch.set_num_threads(1)
    env = None
    try:
        env = make_env()
        model = ModelV4().cpu().eval()
        obs = None; ep_steps = 0; episode_seed = 0; episode_reward = 0.0
        rng = np.random.default_rng((int(master_seed) + 10_000_019 * index) % (2**63))
        while True:
            msg = requests.get()
            if msg is None: break
            generation, state, limit = msg
            model.load_state_dict(state, strict=True)
            rows = {k: [] for k in ('s','j','m','lp','v','rw','done','timeout','timeout_value','mu','std','jump_entropy','dx','death_flag')}
            completed = []
            with torch.inference_mode():
                for _ in range(limit):
                    if obs is None:
                        episode_seed = int(rng.integers(1, 2_147_483_000))
                        torch.manual_seed(episode_seed)
                        obs, info = env.reset(seed=episode_seed)
                        ep_steps = 0; episode_reward = 0.0
                    feature = obs_vector(obs, size=model.obs_dim)
                    x = torch.from_numpy(feature).unsqueeze(0)
                    jump_dist, move_dist, value = model.distributions(x)
                    jump, move = jump_dist.sample(), move_dist.sample()
                    logprob = jump_dist.log_prob(jump) + move_dist.log_prob(move)
                    ji = int(jump.item()); mi = float(move.item())
                    x_before = get_mario_x(obs)
                    next_obs, env_reward, terminated, truncated, info = env.step(env_action(mi, ji))
                    x_after = get_mario_x(next_obs)
                    actual_dx = x_after - x_before if np.isfinite(x_before) and np.isfinite(x_after) else np.nan
                    ep_steps += 1
                    timeout = ep_steps >= max_episode_steps
                    death = bool(terminated and isinstance(info, dict) and info.get('game_state') == 'game_over')
                    # Preserve the established V4 reward baseline; avoid inventing an ungrounded movement bonus.
                    reward = 0.1 - 10.0 * float(death) + (2.0 if terminated and not death else 0.0)
                    episode_reward += reward
                    timeout_value = 0.0
                    if not terminated and (truncated or timeout):
                        next_x = torch.from_numpy(obs_vector(next_obs, size=model.obs_dim)).unsqueeze(0)
                        timeout_value = float(model.act(next_x, deterministic=True)[3].item())
                    for k, v in (('s',feature),('j',ji),('m',mi),('lp',float(logprob.item())),
                                 ('v',float(value.item())),('rw',reward),
                                 ('done',bool(terminated)),('timeout',bool(truncated or timeout)),('timeout_value',timeout_value),
                                 ('mu',float(move_dist.mean.item())), ('std',float(move_dist.stddev.item())),
                                 ('jump_entropy',float(jump_dist.entropy().item())),
                                 ('dx',actual_dx), ('death_flag',int(death))):
                        rows[k].append(v)
                    obs = next_obs
                    if terminated or truncated or timeout:
                        completed.append({'seed': episode_seed, 'steps': ep_steps,
                                          'seconds': float(info.get('time_elapsed', 0)) if isinstance(info, dict) else 0.0,
                                          'reward': episode_reward, 'terminated': bool(terminated)})
                        obs = None
            bootstrap = 0.0
            if obs is not None:
                with torch.inference_mode():
                    bootstrap = float(model.act(torch.from_numpy(obs_vector(obs)).unsqueeze(0), deterministic=True)[3].item())
            result = {k: np.asarray(v, dtype=np.float32 if k in ('s','m','lp','v','rw','timeout_value','mu','std','jump_entropy','dx') else np.int64)
                      for k,v in rows.items()}
            result['bootstrap'] = bootstrap
            result['completed'] = completed
            responses.put((index, generation, result, None))
    except Exception:
        responses.put((index, -1, None, traceback.format_exc()))
    finally:
        if env is not None: env.close()

def gae(roll, gamma, lam):
    rewards, values = roll['rw'], roll['v']
    advantage = np.zeros(len(rewards), dtype=np.float32)
    carry = 0.0
    for t in range(len(rewards)-1, -1, -1):
        terminal = bool(roll['done'][t]); timeout = bool(roll['timeout'][t])
        if t+1 == len(rewards):
            next_v = 0.0 if terminal else float(roll['bootstrap'])
        elif terminal or timeout:
            next_v = 0.0 if terminal else float(roll['timeout_value'][t])
        else:
            next_v = float(values[t+1])
        if timeout and not terminal:
            next_v = float(roll['timeout_value'][t])
        delta = float(rewards[t]) + gamma * next_v - float(values[t])
        carry = delta + gamma * lam * carry if not (terminal or timeout) else delta
        advantage[t] = carry
    return advantage, advantage + values

def update(model, optim, rolls, args, device):
    fields = {key: [] for key in ('s','j','m','lp','adv','ret')}
    for roll in rolls:
        adv, ret = gae(roll, args.gamma, args.gae_lambda)
        for key,val in [('s',roll['s']),('j',roll['j']),('m',roll['m']),('lp',roll['lp']),('adv',adv),('ret',ret)]:
            fields[key].append(val)
    data = {k:torch.as_tensor(np.concatenate(v),device=device) for k,v in fields.items()}
    a = data['adv']; data['adv'] = (a-a.mean())/(a.std(unbiased=False)+1e-8)
    model.train(); losses=[]; clips=[]; kls=[]; actor_losses=[]; value_losses=[]; entropies=[]; grad_norms=[]
    for epoch in range(args.ppo_epochs):
        indices = torch.randperm(len(a), device=device)
        for ix in indices.split(args.minibatch):
            lp, ent, value = model.evaluate_actions(data['s'][ix],data['j'][ix],data['m'][ix])
            ratio = (lp-data['lp'][ix]).exp()
            unclipped = ratio*data['adv'][ix]
            clipped = ratio.clamp(1-args.clip,1+args.clip)*data['adv'][ix]
            actor_loss = -torch.minimum(unclipped,clipped).mean()
            value_loss = (value-data['ret'][ix]).square().mean()
            entropy = ent.mean()
            loss = actor_loss + args.value_coef*value_loss - args.entropy_coef*entropy
            with torch.no_grad():
                # Approximate KL under the behavior policy, independent of clip ratio.
                approx_kl = ((ratio - 1) - (lp-data['lp'][ix])).mean()
            optim.zero_grad(set_to_none=True); loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
            optim.step()
            losses.append(float(loss.detach().cpu()));clips.append(float(((ratio-1).abs()>args.clip).float().mean().detach().cpu()))
            kls.append(float(approx_kl.detach().cpu()))
            actor_losses.append(float(actor_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            grad_norms.append(float(grad_norm.detach().cpu()))
    model.eval()
    movement = np.concatenate([r['m'] for r in rolls]); jumps = np.concatenate([r['j'] for r in rolls])
    move_clip = np.clip(movement,-1,1)
    mus = np.concatenate([r['mu'] for r in rolls]); stds = np.concatenate([r['std'] for r in rolls])
    dxs = np.concatenate([r['dx'] for r in rolls]); valid_dx = dxs[np.isfinite(dxs)]
    jump_entropy = np.concatenate([r['jump_entropy'] for r in rolls])
    n_deaths = sum(int(np.sum(r['death_flag'])) for r in rolls)
    return {'loss': round(float(np.mean(losses)),4),'clip':round(float(np.mean(clips)),4),
            'approx_kl':round(float(np.mean(kls)),7),
            'actor_loss':round(float(np.mean(actor_losses)),5),
            'value_loss':round(float(np.mean(value_losses)),5),
            'entropy':round(float(np.mean(entropies)),5),
            'grad_norm':round(float(np.mean(grad_norms)),5),
            'mu_abs':round(float(np.abs(mus).mean()),5),
            'mu_mean':round(float(mus.mean()),5),
            'std_mean':round(float(stds.mean()),5),
            'jump_entropy':round(float(jump_entropy.mean()),5),
            'actual_dx_abs':round(float(np.abs(valid_dx).mean()),4) if valid_dx.size else None,
            'actual_dx_zero_pct':round(float((np.abs(valid_dx)<1e-6).mean())*100,2) if valid_dx.size else None,
            'dx_available_pct':round(valid_dx.size/len(dxs)*100,2),
            'deaths':n_deaths,
            'move_mean':round(float(move_clip.mean()),3), 'move_abs':round(float(np.abs(move_clip).mean()),3),
            'move_left_pct':round(float((move_clip < -0.15).mean())*100,1),
            'move_right_pct':round(float((move_clip > 0.15).mean())*100,1),
            'move_neutral_pct':round(float((np.abs(move_clip) <= 0.15).mean())*100,1),
            'jump_hist':np.bincount(jumps,minlength=5).tolist()}

def train(args):
    if args.envs < 1 or args.total_steps < 1 or args.batch_steps < 1: raise ValueError('invalid env/step counts')
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    if args.out.resolve() == (args.checkpoint_dir/'model_v4_diagnostic_resume.pt').resolve(): raise ValueError('output collides with resume')
    initial = args.resume if args.resume else args.init
    if not initial.is_file():raise FileNotFoundError(initial)
    ck = torch.load(initial,map_location='cpu',weights_only=False)
    if ck.get('architecture') != 'independent_model_v4':raise ValueError('Model V4 checkpoint required')
    model = ModelV4(int(ck.get('obs_dim',OBS_DIM)),int(ck.get('hidden',HIDDEN))).to(device)
    model.load_state_dict(ck['model'],strict=True)
    optim = torch.optim.Adam(model.parameters(),lr=args.lr,eps=1e-5)
    steps = 0; episodes = 0
    master_seed = int(args.seed if args.seed is not None else secrets.randbelow(2**31-1))
    seed_rng = np.random.default_rng(master_seed)
    if args.resume:
        if ck.get('trainer') not in ('model_v4_500k_diagnostic_v2', 'model_v4_500k_parallel_v1'):raise ValueError('Not a Model V4 step-resume checkpoint')
        if ck.get('envs') != args.envs or ck.get('batch_steps') != args.batch_steps:raise ValueError('Resume worker/batch mismatch')
        steps = int(ck['environment_steps']);episodes = int(ck['completed_episodes'])
        optim.load_state_dict(ck['optimizer'])
        for state in optim.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to(device)
        master_seed = int(ck['master_seed'])
        seed_rng.bit_generator.state=ck['seed_rng']
        print(f'RESUME steps={steps} episodes={episodes} [in-flight game states restart]',flush=True)
    ctx = mp.get_context('spawn')
    input_queues = [ctx.Queue(maxsize=2) for _ in range(args.envs)]
    output_queue = ctx.Queue(maxsize=args.envs*2)
    # Independent random stream per worker; reseed the streams on resume to avoid replaying previous episodes.
    run_seed = int(secrets.randbelow(2**62)) if args.resume else master_seed
    proc = [ctx.Process(target=worker,args=(i,input_queues[i],output_queue,run_seed,args.batch_steps,args.max_episode_steps),daemon=True) for i in range(args.envs)]
    for p in proc:p.start()
    start = time.perf_counter(); wave=0
    def checkpoint():
        payload=make_payload(model,steps=steps,trained=True,seed=master_seed,trainer='model_v4_500k_diagnostic_v2',
                             environment_steps=steps,completed_episodes=episodes,envs=args.envs,batch_steps=args.batch_steps,
                             optimizer=optim.state_dict(),master_seed=master_seed,seed_rng=seed_rng.bit_generator.state)
        atomic_save(payload,args.checkpoint_dir/'model_v4_diagnostic_resume.pt')
    try:
        while steps < args.total_steps:
            remain=args.total_steps-steps
            targets=[min(args.batch_steps,max(0,remain-i*args.batch_steps)) for i in range(args.envs)]
            active=[i for i,n in enumerate(targets) if n>0]
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            for i in active:input_queues[i].put((wave,state,targets[i]))
            collected={}
            while len(collected)<len(active):
                try: i,g,result,error=output_queue.get(timeout=10)
                except queue.Empty:
                    if any(not proc[i].is_alive() for i in active):raise RuntimeError('Worker terminated unexpectedly')
                    continue
                if error: raise RuntimeError(f'Worker {i}: {error}')
                if g!=wave:raise RuntimeError(f'Wave mismatch expected={wave} got={g}')
                collected[i]=result
            rolls=[collected[i] for i in active]
            stats=update(model,optim,rolls,args,device)
            steps+=sum(len(r['rw']) for r in rolls)
            records=[ep for r in rolls for ep in r['completed']]
            episodes+=len(records)
            wave+=1
            log={'wave':wave,'steps':steps,'target':args.total_steps,'episodes':episodes,
                 'mean_ended_s':round(float(np.mean([ep['seconds'] for ep in records])),2) if records else None,
                 'wall_s':round(time.perf_counter()-start,1),**stats}
            print('UPDATE '+json.dumps(log,ensure_ascii=False),flush=True)
            if steps % args.checkpoint_every < len(active)*args.batch_steps or steps>=args.total_steps:
                checkpoint()
                print(f'CHECKPOINT {steps} {args.checkpoint_dir / "model_v4_diagnostic_resume.pt"}',flush=True)
        atomic_save(make_payload(model,steps=steps,trained=True,seed=master_seed,trainer='model_v4_500k_diagnostic_v2',completed_episodes=episodes),args.out)
        print(f'FINAL {args.out} steps={steps} episodes={episodes}',flush=True)
    finally:
        for q in input_queues:
            try:q.put_nowait(None)
            except queue.Full:pass
        for p in proc:
            p.join(timeout=1)
            if p.is_alive():p.terminate();p.join(timeout=1)
        for q in input_queues:q.close()
        output_queue.close()

def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Model V4 500k step PPO; 16 Windows spawn workers')
    p.add_argument('--init',type=Path,default=HERE/'model_v4_initial_50k.pt')
    p.add_argument('--resume',type=Path)
    p.add_argument('--total-steps',type=int,default=16384)
    p.add_argument('--envs',type=int,default=16)
    p.add_argument('--batch-steps',type=int,default=256)
    p.add_argument('--max-episode-steps',type=int,default=9000)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    p.add_argument('--gamma',type=float,default=0.995)
    p.add_argument('--gae-lambda',type=float,default=0.95)
    p.add_argument('--ppo-epochs',type=int,default=2)
    p.add_argument('--minibatch',type=int,default=1024)
    p.add_argument('--clip',type=float,default=0.2)
    p.add_argument('--value-coef',type=float,default=0.5)
    p.add_argument('--entropy-coef',type=float,default=0.02)
    p.add_argument('--max-grad-norm',type=float,default=0.5)
    p.add_argument('--checkpoint-every',type=int,default=16384)
    p.add_argument('--checkpoint-dir',type=Path,default=HERE/'checkpoints')
    p.add_argument('--out',type=Path,default=HERE/'model_v4_diagnostic.pt')
    p.add_argument('--seed',type=int)
    return p.parse_args(argv)

if __name__=='__main__':
    mp.freeze_support()
    train(parse_args())
