"""
정확한 시뮬레이터(exact_sim.py) 검증
  python validate_exact.py --module agent_ppo_v5 --model models/bullet_bill_ppo_v5_long_best.pt

1) 킬러 생성 스케줄: 예측한 생성 프레임이 속한 스텝에 실제로 새 킬러가 화면 끝에 나타나는지
2) 궤적: 에이전트 플레이 중 고른 상태에서 무작위 행동 24스텝을 환경과 시뮬레이터에 똑같이 넣고
        매 스텝 마리오 좌표(left, top)와 사고 시점이 정확히 같은지 비교
        (지면 상태와 공중 상태, 가운데 땅 밖 상태를 모두 포함)
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import importlib
import sys

import gymnasium as gym
import kymnasium as kym
import numpy as np

import exact_sim as E

_mod = sys.argv[sys.argv.index('--module') + 1] if '--module' in sys.argv else 'agent_ppo_v5'
_M = importlib.import_module(_mod)
ENV_ID = 'kymnasium/BulletBill-Continuous-Hard-Stage-2'


def check_spawn(agent, env, episodes=10):
    hit = miss = extra = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=50_000 + ep)
        agent.fe.reset()
        step = 0
        while True:
            obs, _, term, trunc, info = env.step(agent.act(obs, info))
            step += 1
            lo, hi = 4 * (step - 1), 4 * step
            predicted = np.any((E.SPAWN_FRAMES > lo) & (E.SPAWN_FRAMES <= hi))
            bb = obs['bullet_bills']
            bb = bb[np.abs(bb).sum(1) > 0]
            # 새 킬러 = 생성된 쪽 끝에서 안쪽으로 막 들어오는 중 (반대편으로 빠져나가는 킬러 제외)
            fresh = np.any(((bb[:, 0] < -48 + 25) & (bb[:, 4] > 0)) |
                           ((bb[:, 0] > 1296 - 25) & (bb[:, 4] < 0))) if len(bb) else False
            n_active = len(bb)
            if predicted and fresh:
                hit += 1
            elif predicted and not fresh and n_active < 30:
                miss += 1
            elif fresh and not predicted:
                extra += 1
            if term or trunc:
                break
    print(f"[1] 생성 스케줄: 예측 일치 {hit}, 예측했는데 새 킬러 없음 {miss}, 예측 없는데 새 킬러 {extra}")


def check_traj(agent, env, episodes=30, per_ep=6, seq_len=24):
    rng = np.random.default_rng(0)
    n = exact_pos = exact_death = 0
    first_bad = []
    stats = {'ground': [0, 0], 'air': [0, 0], 'off_center': [0, 0]}
    for ep in range(episodes):
        seed = 60_000 + ep
        obs, info = env.reset(seed=seed)
        agent.fe.reset()
        actions, observations = [], [obs]
        while True:
            a = agent.act(obs, info)
            obs, _, term, trunc, info = env.step(a)
            actions.append(a)
            observations.append(obs)
            if term or trunc:
                break
        if len(actions) < 6:
            continue
        tracker = E.VyTracker()
        vys = [tracker.update(o_['mario']) for o_ in observations]      # 에피소드를 따라가며 낙하 속도 복원
        ks = rng.choice(np.arange(2, len(actions) - 1), size=min(per_ep, len(actions) - 3), replace=False)
        for k in ks:
            o = observations[k]
            if o['mario'][6] > 0.5:                     # 무적 상태는 남은 시간을 모르므로 제외
                continue
            # 무작위 행동 열: 이동은 -1~1 연속, 점프는 70% 확률로 0, 아니면 0~1
            seq = np.stack([rng.uniform(-1, 1, seq_len),
                            np.where(rng.random(seq_len) < 0.7, 0.0, rng.uniform(0, 1, seq_len))], axis=1)
            mario, bb, st, inv = E.obs_to_sim(o, vy=vys[k])
            traj, f_sim, c_sim = E.simulate_trajectory(mario, bb, st, inv, seq, E.SOLID)
            death_step_sim = (f_sim + 3) // 4 if f_sim > 0 else -1
            # 환경에서 같은 상태 재현 후 같은 행동 열
            env.reset(seed=seed)
            for a in actions[:k]:
                env.step(a)
            pos_ok, death_step_env = True, -1
            for s in range(seq_len):
                o2, _, term, trunc, inf = env.step(seq[s].astype(np.float32))
                if term or trunc:
                    if inf.get('game_state') == 'game_over':
                        death_step_env = s + 1
                    break
                if death_step_sim != -1 and s + 1 >= death_step_sim:
                    pos_ok = False
                    break
                if not (o2['mario'][0] == traj[s, 0] and o2['mario'][1] == traj[s, 1]):
                    pos_ok = False
                    if len(first_bad) < 5:
                        first_bad.append((seed, int(k), s + 1, o2['mario'][:2].tolist(), traj[s, :2].tolist()))
                    break
            death_ok = death_step_env == death_step_sim
            n += 1
            exact_pos += int(pos_ok or (death_ok and death_step_env != -1))
            exact_death += int(death_ok)
            m = o['mario']
            key = 'off_center' if not (528 <= (m[0] + m[2]) / 2 <= 768) else ('air' if m[7] > 0.5 else 'ground')
            stats[key][0] += 1
            stats[key][1] += int(death_ok and (pos_ok or death_step_env != -1))
        print(f"  {ep + 1}/{episodes} 판 완료", end='\r')
    print(f"\n[2] 궤적 대조 {n}건: 매 스텝 좌표 완전 일치 {exact_pos / n * 100:.1f}%, "
          f"사고 시점 일치 {exact_death / n * 100:.1f}%")
    for k, (tot, ok) in stats.items():
        if tot:
            print(f"    {k:10s}: {ok}/{tot} 일치")
    for b in first_bad:
        print(f"    불일치 예: seed {b[0]}, 상태 step {b[1]}, {b[2]}스텝째 환경 {b[3]} vs 시뮬레이터 {b[4]}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--module', default='agent_ppo_v5')
    p.add_argument('--model', default='models/bullet_bill_ppo_v5_long_best.pt')
    args = p.parse_args()
    agent = _M.BulletBillAgent.load(args.model)
    env = gym.make(id=ENV_ID, render_mode='none', bgm=False)
    print(f"numba: {E.HAS_NUMBA}")
    check_spawn(agent, env)
    check_traj(agent, env)
    env.close()
