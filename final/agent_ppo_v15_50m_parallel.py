"""Bullet Bills Stage-2: single-file PPO + DAgger agent and parallel trainer.

Official environment is never mutated. 16 independent CPU workers collect rollouts;
optional CUDA learner updates actor/critic. 50M steps, no interim evaluation.
Automatic recovery checkpointing is not an evaluation or model-selection procedure.

Examples:
  python agent_ppo_v15_50m_parallel.py --steps 50000000 --envs 16 --device cuda
  python agent_ppo_v15_50m_parallel.py --resume models/parallel50m_resume.pt --steps 50000000
  python agent_ppo_v15_50m_parallel.py --eval 200 --model models/parallel50m_final.pt
  python agent_ppo_v15_50m_parallel.py --demo --model models/parallel50m_final.pt --seed 42
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')   # numpy(OpenBLAS) 시작 시 메모리 할당 실패 방지
import argparse
import math
import os
import random
import time
from collections import deque
from typing import Any, Dict

import gymnasium as gym
import kymnasium as kym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

try:                                   # 시뮬레이션 가속 (없으면 순수 파이썬 반복문으로 실행)
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        return lambda f: f

ENV_ID = 'kymnasium/BulletBill-Continuous-Hard-Stage-2'

# ================= 측정한 환경 상수 =================
DT = 1.0 / 15.0
GROUND = 768.0                      # 서 있을 때 마리오 bottom
PLAT_C, PLAT_HALF = 648.0, 144.0    # 마리오 중심 기준 발판 504 ~ 792
EDGE_WARN = 48.0                    # 발판 끝 48px 안쪽부터 위험
DECAY = 0.656                       # 손 뗐을 때 스텝당 속도 감쇠
GRAVITY = 1800.0
JUMP_V = 900.0
MAX_JUMP_H = 226.0
SIZE = 48.0                         # 마리오, 킬러 크기
GROUND_LANE_TOP = 672.0             # 킬러 top > 672 이면 서 있는 마리오와 겹침
STOMPABLE_TOP = 542.0               # 최대 점프 시 마리오 bottom 높이
VX_SCALE, BB_VX_SCALE = 650.0, 300.0
TTC_MAX = 3.0                       # 충돌까지 시간 상한(초)
INV_SCALE = 150.0                   # 무적 경과 스텝 정규화 (지속 시간 미확인)

K_BULLETS, BULLET_FEAT, MARIO_FEAT, N_STARS, STAR_FEAT = 8, 10, 14, 2, 3
SIM_OPTS = np.array([0.0, 0.5, 0.6, 0.75, 1.0])   # 0 = 안 뜀
SIM_STEPS = 24                                     # 1.6초 앞까지
SIM_MARGIN, STOMP_BOUNCE, STOMP_TOL = 2.0, -500.0, 4.0
MOVE_ACC = 309.5                                   # 측정: vx' = 0.656 * vx + 309.5 * 이동 힘
PLAN_MOVES = [(0.0, 0), (1.0, 3), (1.0, 6), (-1.0, 3), (-1.0, 6)]
PLAN_JUMPS = [(0, 0.0)] + [(d, f) for d in (0, 2, 4, 6, 8, 10, 12) for f in (0.5, 0.6, 0.75, 1.0)]
PLANS = np.array([(mf, nm, d, f) for (mf, nm) in PLAN_MOVES for (d, f) in PLAN_JUMPS], dtype=np.float64)
_NJ = len(PLAN_JUMPS)
NOW_IDX = np.arange(5)                                                   # 이동 없음, 즉시: 안 뜀 + 4높이
WAIT_IDX = [np.arange(1 + di * 4, 5 + di * 4) for di in range(1, 7)]     # 이동 없음, 2~12스텝 지연
MOVE_IDX = [np.arange(mi * _NJ, (mi + 1) * _NJ) for mi in range(1, 5)]   # 이동 4가지 (점프 계획 포함)
PLAN_FEAT = len(NOW_IDX) + len(WAIT_IDX) + len(MOVE_IDX)                 # 15
OBS_DIM = MARIO_FEAT + PLAN_FEAT + 3 + K_BULLETS * BULLET_FEAT + N_STARS * STAR_FEAT + 30 + 30   # 178 (v12: 조합 점수 30)
ACT_DIM = 2

# ================= 보상 =================
SURVIVE, DEATH, JUMP_COST = 0.1, -10.0, 0.0     # v15: 점프 비용 제거 (v14까지 -0.02)
SHAPE, BULLET_W = 0.3, 1.0          # v6: 킬러 위험 = 계획 기반 위험도
STAR_EARLY_PENALTY, STAR_FREE_TIME = 5.0, 60.0   # v9: 별을 먹는 순간 -5 x max(0, 1 - t/60)
GAMMA = 0.995          # v10: 0.99 -> 0.995 (약 13초 앞을 중시)

# ================= 행동 =================
JUMP_LEVELS = np.array([0.0, 0.5, 0.6, 0.75, 1.0], dtype=np.float32)   # 안 뜀, 57/81/127/226px
N_JUMP = len(JUMP_LEVELS)


N_ACT = N_JUMP * 3                                  # v14: 15가지 조합, 번호 = 점프 x 3 + (이동 + 1)
PAIR_OFF = 148                                      # v14: 상태 안에서 실제 킬러 기준 조합 점수의 시작 위치


def pair_to_env(k) -> np.ndarray:
    """v14: 조합 번호 -> 환경 행동 [이동 힘(-1/0/+1), 점프 힘]"""
    j, mv = divmod(int(k), 3)
    return np.array([float(mv - 1), float(JUMP_LEVELS[j])], dtype=np.float32)


def to_env_action(move, jump_idx) -> np.ndarray:
    return np.array([float(np.clip(move, -1.0, 1.0)), float(JUMP_LEVELS[int(jump_idx)])], dtype=np.float32)


def simulate_np(m, bb, inv):
    """[numpy 버전] 점프 선택지별 첫 사고 시각(초, 없으면 inf). 각 선택지는 '첫 스텝에 점프 f, 이후 입력 없음'.
    보정된 가정: 공중 수평 감쇠 0.656, 착지/밟기 순간 점프 실행, 판정 축소 2px."""
    n_opt, nb = len(SIM_OPTS), len(bb)
    air0 = m[7] > 0.5
    x = np.full(n_opt, (m[0] + m[2]) / 2, dtype=np.float64)
    bot = np.full(n_opt, float(m[3]))
    vx = np.full(n_opt, float(m[4]))
    if air0:
        vy = np.full(n_opt, float(m[5]))
        air = np.ones(n_opt, bool)
    else:
        vy = -JUMP_V * SIM_OPTS
        air = SIM_OPTS > 0
    first = np.full(n_opt, np.inf)
    b_alive = np.ones((n_opt, nb), bool)
    for k in range(1, SIM_STEPS + 1):
        tk = k * DT
        prev_bot = bot.copy()
        bot = np.where(air, bot + vy * DT + 0.5 * GRAVITY * DT * DT, bot)
        vy = np.where(air, vy + GRAVITY * DT, 0.0)
        vx = vx * DECAY
        x = x + vx * DT
        on_plat = np.abs(x - PLAT_C) <= PLAT_HALF
        land = air & (vy > 0) & (bot >= GROUND) & on_plat & (prev_bot <= GROUND + 1)
        bot = np.where(land, GROUND, bot)
        vy = np.where(land, 0.0, vy)
        air = air & ~land
        if k == 1 and air0:                        # 착지 순간 점프 실행
            lj = land & (SIM_OPTS > 0)
            vy = np.where(lj, -JUMP_V * SIM_OPTS, vy)
            air = air | lj
        air = air | (~on_plat & ~air)
        first = np.where((bot > GROUND + 30) & np.isinf(first), tk, first)
        if nb and not inv:
            bl = bb[:, 0] + bb[:, 4] * tk
            ovx = (x[:, None] - SIZE / 2 < bl + SIZE - SIM_MARGIN) & (x[:, None] + SIZE / 2 > bl + SIM_MARGIN)
            ovy = (bot[:, None] - SIZE < bb[:, 3] - SIM_MARGIN) & (bot[:, None] > bb[:, 1] + SIM_MARGIN)
            stomp = (vy[:, None] > 0) & (prev_bot[:, None] <= bb[:, 1] + STOMP_TOL)
            touch = ovx & ovy & b_alive
            stomped, hit = touch & stomp, touch & ~stomp
            if stomped.any():
                b_alive &= ~stomped
                s_opt = stomped.any(1)
                new_vy = np.where(SIM_OPTS > 0, -JUMP_V * SIM_OPTS, STOMP_BOUNCE) if k == 1 \
                    else np.full(n_opt, STOMP_BOUNCE)
                vy = np.where(s_opt, new_vy, vy)
                air = air | s_opt
            first = np.where(hit.any(1) & np.isinf(first), tk, first)
        if np.isfinite(first).all():               # 모든 선택지의 결과가 정해지면 종료
            break
    return first


@njit(cache=True)
def _simulate_loop(m, bb, inv, opts, dt, ground, plat_c, plat_half, decay, gravity, jump_v, size,
                   margin, bounce, stomp_tol, n_steps):
    """[반복문 버전, numba로 컴파일] simulate_np와 같은 계산을 선택지별로 독립 수행"""
    nb = bb.shape[0]
    out = np.full(opts.shape[0], np.inf)
    air0 = m[7] > 0.5
    for o in range(opts.shape[0]):
        f = opts[o]
        x = (m[0] + m[2]) / 2.0
        bot = m[3]
        vx = m[4]
        if air0:
            vy = m[5]
            air = True
        else:
            vy = -jump_v * f
            air = f > 0
        alive = np.ones(nb, dtype=np.bool_)
        first = np.inf
        for k in range(1, n_steps + 1):
            tk = k * dt
            prev_bot = bot
            if air:
                bot = bot + vy * dt + 0.5 * gravity * dt * dt
                vy = vy + gravity * dt
            else:
                vy = 0.0
            vx = vx * decay
            x = x + vx * dt
            on_plat = abs(x - plat_c) <= plat_half
            if air and vy > 0 and bot >= ground and on_plat and prev_bot <= ground + 1:
                bot = ground
                vy = 0.0
                air = False
                if k == 1 and air0 and f > 0:      # 착지 순간 점프 실행
                    vy = -jump_v * f
                    air = True
            if not on_plat:
                air = True
            if bot > ground + 30 and first == np.inf:
                first = tk
            if nb > 0 and not inv:
                any_stomp = False
                any_hit = False
                for j in range(nb):
                    if not alive[j]:
                        continue
                    bl = bb[j, 0] + bb[j, 4] * tk
                    if (x - size / 2 < bl + size - margin) and (x + size / 2 > bl + margin) and \
                            (bot - size < bb[j, 3] - margin) and (bot > bb[j, 1] + margin):
                        if vy > 0 and prev_bot <= bb[j, 1] + stomp_tol:
                            alive[j] = False
                            any_stomp = True
                        else:
                            any_hit = True
                if any_stomp:
                    vy = -jump_v * f if (k == 1 and f > 0) else bounce
                    air = True
                if any_hit and first == np.inf:
                    first = tk
            if first != np.inf:
                break
        out[o] = first
    return out


def simulate(m, bb, inv):
    """점프 선택지별 첫 사고 시각(초, 없으면 inf). 반복문 버전 사용 (numba가 있으면 컴파일되어 더 빠름).
    numba가 없어도 선택지별로 일찍 끝나는 반복문이 numpy 버전보다 빠름."""
    return _simulate_loop(np.asarray(m, np.float64), np.ascontiguousarray(bb, np.float64), bool(inv),
                          SIM_OPTS.astype(np.float64), DT, GROUND, PLAT_C, PLAT_HALF, DECAY, GRAVITY, JUMP_V,
                          SIZE, SIM_MARGIN, STOMP_BOUNCE, STOMP_TOL, SIM_STEPS)


@njit(cache=True)
def plan_sim(m, bb, inv, plans, dt, ground, plat_c, plat_half, decay, acc, gravity, jump_v, size,
             margin, bounce, stomp_tol, n_steps):
    nb = bb.shape[0]
    out = np.full(plans.shape[0], np.inf)
    for p in range(plans.shape[0]):
        mf, nm, d, f = plans[p, 0], plans[p, 1], plans[p, 2], plans[p, 3]
        x = (m[0] + m[2]) / 2.0
        bot = m[3]
        vx = m[4]
        air = m[7] > 0.5
        vy = m[5] if air else 0.0
        alive = np.ones(nb, dtype=np.bool_)
        first = np.inf
        for k in range(1, n_steps + 1):
            tk = k * dt
            jump_now = f > 0 and k == d + 1
            if jump_now and not air:                 # 지면에서 점프
                vy = -jump_v * f
                air = True
            prev_bot = bot
            if air:
                bot = bot + vy * dt + 0.5 * gravity * dt * dt
                vy = vy + gravity * dt
            else:
                vy = 0.0
            mv = mf if k <= nm else 0.0
            vx = vx * decay + acc * mv
            x = x + vx * dt
            on_plat = abs(x - plat_c) <= plat_half
            if air and vy > 0 and bot >= ground and on_plat and prev_bot <= ground + 1:
                bot = ground
                vy = 0.0
                air = False
                if jump_now:                         # 착지 순간 점프 실행
                    vy = -jump_v * f
                    air = True
            if not on_plat:
                air = True
            if bot > ground + 30 and first == np.inf:
                first = tk
            if nb > 0 and not inv:
                any_stomp = False
                any_hit = False
                for j in range(nb):
                    if not alive[j]:
                        continue
                    bl = bb[j, 0] + bb[j, 4] * tk
                    if (x - size / 2 < bl + size - margin) and (x + size / 2 > bl + margin) and \
                            (bot - size < bb[j, 3] - margin) and (bot > bb[j, 1] + margin):
                        if vy > 0 and prev_bot <= bb[j, 1] + stomp_tol:
                            alive[j] = False
                            any_stomp = True
                        else:
                            any_hit = True
                if any_stomp:
                    vy = -jump_v * f if jump_now else bounce
                    air = True
                if any_hit and first == np.inf:
                    first = tk
            if first != np.inf:
                break
        out[p] = first
    return out


# ===================== v7: 정확한 시뮬레이터 (환경 소스 규칙을 프레임 단위로) =====================
# ---------- 코드 상수 (consts.py) ----------
FPS, FRAMES_PER_STEP = 60, 4
FRAME_DT = 1.0 / FPS
TILE, MAP_W, MAP_H = 48, 27, 18
SCREEN_W, SCREEN_H = TILE * MAP_W, TILE * MAP_H          # 1296, 864
MARIO_W = MARIO_H = 48
GROUND_MAX_SPEED, AIR_MAX_SPEED, IDLE_THRESHOLD = 750.0, 500.0, 10.0
FRICTION, EXACT_GRAVITY = 0.90, 2000.0   # 기존 v1 특징의 GRAVITY(측정 근사 1800)와 구분
MAX_MOVING_FORCE, MAX_FLOATING_FORCE = 100.0, 75.0
JUMPING_FORCE, HOP_FORCE = 1000.0, 500.0
INVINCIBLE_TIME = 10.0

# ---------- 맵 (stage-2, map_ascii.py로 확인한 블록 79개) ----------
BLOCKS = [(3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (19, 3), (20, 3), (21, 3), (22, 3), (23, 3),
          (11, 6), (12, 6), (13, 6), (14, 6), (15, 6),
          (0, 9), (1, 9), (2, 9), (3, 9), (7, 9), (8, 9), (9, 9), (10, 9),
          (16, 9), (17, 9), (18, 9), (19, 9), (23, 9), (24, 9), (25, 9), (26, 9),
          (5, 13), (21, 13), (4, 14), (5, 14), (6, 14), (20, 14), (21, 14), (22, 14),
          (3, 15), (4, 15), (5, 15), (6, 15), (7, 15), (19, 15), (20, 15), (21, 15), (22, 15), (23, 15)]
BLOCKS += [(c, r) for r in (16, 17) for c in list(range(3, 8)) + list(range(11, 16)) + list(range(19, 24))]
SOLID = np.zeros((MAP_H, MAP_W), dtype=np.bool_)
for _c, _r in BLOCKS:
    SOLID[_r, _c] = True
assert SOLID.sum() == 79


# ---------- 킬러 생성 스케줄 (env.py _step_internal 그대로) ----------
def spawn_frames(game_duration=180.0, init_interval=1.5, min_interval=0.1, max_spawn_duration=105.0):
    dt, t, timer, interval = 1.0 / FPS, 0.0, 0.0, init_interval
    frames, f = [], 0
    while True:
        f += 1
        timer += dt
        t += dt
        if t >= game_duration:
            break
        if timer >= interval:
            timer = 0.0
            frames.append(f)
        interval = init_interval * (min_interval / init_interval) ** (t / max_spawn_duration)
    return np.array(frames, dtype=np.int64)


TRAIN_DURATION = 180.0  # 공식 게임 제한을 변경하지 않음
SPAWN_FRAMES = spawn_frames(game_duration=TRAIN_DURATION)


def frames_to_next_spawns(frame, k=2):
    """현재 프레임 이후 k개 생성까지 남은 프레임 수"""
    i = np.searchsorted(SPAWN_FRAMES, frame, side='right')
    out = np.full(k, 10_000, dtype=np.int64)
    nxt = SPAWN_FRAMES[i:i + k] - frame
    out[:len(nxt)] = nxt
    return out


# ---------- 프레임 단위 시뮬레이션 ----------
@njit(cache=True)
def _overlap(al, at, ar, ab, bl, bt, br, bb):
    return al < br and ar > bl and at < bb and ab > bt


@njit(cache=True)
def _run(left0, top0, vx0, vy0, grounded0, inv0, bul, stars, act, solid, traj, record):
    """한 계획 실행. 반환: (사고 프레임(없으면 -1), 원인 1=킬러 2=낙하)"""
    left, top, vx, vy, grounded, inv = left0, top0, vx0, vy0, grounded0, inv0
    dt = 1.0 / 60.0
    nb = bul.shape[0]
    bl = np.empty(nb)
    bt = np.empty(nb)
    bv = np.empty(nb)
    balive = np.ones(nb, dtype=np.bool_)
    for j in range(nb):
        bl[j], bt[j], bv[j] = bul[j, 0], bul[j, 1], bul[j, 4]
    ns = stars.shape[0]
    salive = np.ones(ns, dtype=np.bool_)
    n_steps = act.shape[0]
    frame = 0
    for s in range(n_steps):
        fx_in = act[s, 0] * 100.0
        fy_in = -act[s, 1] * 1000.0
        for _ in range(4):
            frame += 1
            # ---- apply_force ----
            if grounded:
                fx = max(-100.0, min(100.0, fx_in))
            else:
                fx = max(-75.0, min(75.0, fx_in))
            vx += fx
            if grounded and fy_in < 0:
                vy = max(-1000.0, min(0.0, fy_in))
                grounded = False
            # ---- 주변 블록 범위 (tick 이전 위치 기준, env._nearby_platforms) ----
            c0 = max(0, int(left) // 48 - 2)
            c1 = min(26, int(left + 48) // 48 + 2)
            r0 = max(0, int(top) // 48 - 2)
            r1 = min(17, int(top + 48) // 48 + 2)
            # ---- tick_frame ----
            if grounded:
                vx = max(-750.0, min(750.0, vx))
            else:
                vx = max(-500.0, min(500.0, vx))
            vy += 2000.0 * dt
            if abs(vx) < 10.0:
                vx = 0.0
            left += float(int(vx * dt))
            if left < 0:
                left = 0.0
            if left + 48 > 1296:
                left = 1296.0 - 48
            vx *= 0.90
            # 겹칠 수 있는 칸만 같은 순서(x 우선)로 검사 (env는 주변 5x5칸을 모두 검사하지만 결과는 같음)
            ca, cb = max(c0, int(left) // 48), min(c1, int(left + 47) // 48)
            ra, rb = max(r0, int(top) // 48), min(r1, int(top + 47) // 48)
            for gx in range(ca, cb + 1):
                for gy in range(ra, rb + 1):
                    if solid[gy, gx]:
                        tl, tt = gx * 48.0, gy * 48.0
                        if _overlap(left, top, left + 48, top + 48, tl, tt, tl + 48, tt + 48):
                            if vx > 0:
                                left = tl - 48
                            elif vx < 0:
                                left = tl + 48
            top += float(int(vy * dt))
            grounded = False
            ca, cb = max(c0, int(left) // 48), min(c1, int(left + 47) // 48)
            ra, rb = max(r0, int(top) // 48), min(r1, int(top + 47) // 48)
            for gx in range(ca, cb + 1):
                for gy in range(ra, rb + 1):
                    if solid[gy, gx]:
                        tl, tt = gx * 48.0, gy * 48.0
                        if _overlap(left, top, left + 48, top + 48, tl, tt, tl + 48, tt + 48):
                            if vy > 0:
                                top = tt - 48
                                vy = 0.0
                                grounded = True
                            elif vy < 0:
                                top = tt + 48
                                vy = 0.0
            if not grounded and vy >= 0:
                sup = False
                gy_below = int(top + 48) // 48              # bottom == 블록 top 인 줄만 해당
                for gx in range(max(c0, int(left) // 48), min(c1, int(left + 47) // 48) + 1):
                    for gy in range(max(r0, gy_below), min(r1, gy_below) + 1):
                        if solid[gy, gx]:
                            tl, tt = gx * 48.0, gy * 48.0
                            if top + 48 == tt and left + 48 > tl and left < tl + 48:
                                sup = True
                grounded = sup
                if sup:
                    vy = 0.0
            if inv > 0:
                inv -= 1
            if top + 48 > 864:
                return frame, 2
            # ---- 킬러 이동 ----
            for j in range(nb):
                if balive[j]:
                    bl[j] += float(int(bv[j] * dt))
                    if (bv[j] > 0 and bl[j] > 1296) or (bv[j] < 0 and bl[j] + 48 < 0):
                        balive[j] = False
            # ---- 별 ----
            for k in range(ns):
                if salive[k] and _overlap(left, top, left + 48, top + 48,
                                          stars[k, 0], stars[k, 1], stars[k, 2], stars[k, 3]):
                    inv += 600
                    salive[k] = False
                    break
            # ---- 킬러 충돌 ----
            for j in range(nb):
                if balive[j] and _overlap(left, top, left + 48, top + 48, bl[j], bt[j], bl[j] + 48, bt[j] + 48):
                    if vy > 0 and top + 48 < bt[j] + 24:
                        vy = -500.0
                        balive[j] = False
                    elif inv > 0:
                        balive[j] = False
                    else:
                        return frame, 1
        if record:
            traj[s, 0], traj[s, 1], traj[s, 2], traj[s, 3], traj[s, 4] = left, top, vx, vy, 1.0 if grounded else 0.0
    return -1, 0


@njit(cache=True)
def simulate_plans(mario, bullets, stars, inv_frames, actions, solid):
    """mario: [left, top, vx, vy, grounded], bullets: (n,5), stars: (k,4), actions: (P,S,2)
    반환: 사고 프레임 (P,) (-1 = 없음), 원인 (P,)"""
    P = actions.shape[0]
    out_f = np.full(P, -1, dtype=np.int64)
    out_c = np.zeros(P, dtype=np.int64)
    dummy = np.zeros((1, 5))
    for p in range(P):
        f, c = _run(mario[0], mario[1], mario[2], mario[3], mario[4] > 0.5, inv_frames,
                    bullets, stars, actions[p], solid, dummy, False)
        out_f[p] = f
        out_c[p] = c
    return out_f, out_c


@njit(cache=True)
def simulate_trajectory(mario, bullets, stars, inv_frames, act, solid):
    """한 행동 열의 스텝별 마리오 상태 (S,5) [left, top, vx, vy, grounded]와 사고 프레임·원인"""
    traj = np.full((act.shape[0], 5), np.nan)
    f, c = _run(mario[0], mario[1], mario[2], mario[3], mario[4] > 0.5, inv_frames,
                bullets, stars, act, solid, traj, True)
    return traj, f, c



# 계획 145개를 스텝별 행동 배열로: 이동 mf를 nm스텝 유지, d번째 스텝에 점프 f (그 스텝 4프레임 동안 입력)
PLAN_ACTIONS = np.zeros((len(PLANS), SIM_STEPS, 2), dtype=np.float64)
for _p, (_mf, _nm, _d, _f) in enumerate(PLANS):
    PLAN_ACTIONS[_p, :int(_nm), 0] = _mf
    if _f > 0:
        PLAN_ACTIONS[_p, int(_d), 1] = _f
SPAWN_FEAT = 3


def spawn_features(frame, t):
    """다음 두 번의 생성까지 남은 시간(1.5초로 정규화, 1에서 자름), 현재 생성 간격(/1.5)"""
    nxt = frames_to_next_spawns(frame, 2) / 90.0
    interval = 1.5 * (0.1 / 1.5) ** (t / 105.0)
    return np.array([min(nxt[0], 1.0), min(nxt[1], 1.0), interval / 1.5], dtype=np.float32)


_VY_STEP = 2000.0 * (1.0 / 60.0)
_VY_STARTS = (0.0, -500.0, -600.0, -750.0, -1000.0)


def restore_vy(vy_obs):
    """관측값(32비트)으로 잘린 낙하 속도를 환경의 64비트 값으로 복원 (시작값 + 33.33... x 프레임 수 역산)"""
    target = np.float32(vy_obs)
    for v0 in _VY_STARTS:
        k = int(round((float(vy_obs) - v0) / _VY_STEP))
        if 0 <= k <= 200:
            v = v0
            for _ in range(k):
                v += _VY_STEP
            if np.float32(v) == target:
                return v
    return float(vy_obs)


def _accum(v0, k):
    v = v0
    for _ in range(k):
        v += _VY_STEP
    return v


class VyTracker:
    """에피소드 동안 낙하 속도의 '시작값과 프레임 수'를 이어서 추적해 64비트 값을 복원.
    공중에서는 스텝마다 4프레임씩 더해지므로 직전 값에서 이어 가면 32비트로는 구분 안 되는 경우도 풀림."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.v0, self.k = None, 0

    def update(self, m):
        if m[7] < 0.5:                                  # 지면: 속도 0, 추적 초기화
            self.reset()
            return float(m[5])
        target = np.float32(m[5])
        if self.v0 is not None:                         # 직전 상태에서 이어 가기 (보통 +4프레임)
            for dk in (4, 3, 2, 1, 5, 6, 7, 8):
                v = _accum(self.v0, self.k + dk)
                if np.float32(v) == target:
                    self.k += dk
                    return v
        # 새로 찾기: 지면에서 스텝 첫 프레임에 점프하면 정확히 4프레임 진행(k=4)이므로 우선,
        # 그다음은 가장 최근 사건(프레임 수가 가장 작은 것: 밟기 반동, 착지 순간 점프, 낙하 시작)
        best = None
        for v0 in _VY_STARTS:
            k = int(round((float(m[5]) - v0) / _VY_STEP))
            if 0 <= k <= 200:
                v = _accum(v0, k)
                if np.float32(v) == target:
                    key = (k != 4, k)
                    if best is None or key < best[3]:
                        best = (v0, k, v, key)
        if best is not None:
            self.v0, self.k = best[0], best[1]
            return best[2]
        self.reset()
        return float(m[5])


def plan_features(m, bb, inv_frames, stars, vy=None):
    """계획 145개를 정확한 시뮬레이터로 계산해 15차원으로 요약 (값: 첫 사고 시각 / 1.6초, 1 = 사고 없음)
    vy: VyTracker로 복원한 낙하 속도 (없으면 한 시점 복원)"""
    if vy is None:
        vy = restore_vy(m[5]) if m[7] > 0.5 else float(m[5])
    mario = np.array([m[0], m[1], m[4], vy, 0.0 if m[7] > 0.5 else 1.0], dtype=np.float64)
    f, _ = simulate_plans(mario, np.ascontiguousarray(bb, np.float64), np.ascontiguousarray(stars, np.float64),
                          int(inv_frames), PLAN_ACTIONS, SOLID)
    horizon = SIM_STEPS * DT
    first = np.where(f > 0, f / 60.0, np.inf)
    t = np.minimum(first, horizon) / horizon
    feat = np.concatenate([t[NOW_IDX], [t[i].max() for i in WAIT_IDX],
                           [t[i].max() for i in MOVE_IDX]]).astype(np.float32)
    return feat, float(t.max())          # (15차원 특징, 최선 계획의 안전도 0~1)


# ===================== v10: 2단계 생존 예측 + 미래 킬러 생성 예측 =====================
GMAX = 80                                     # 가상 킬러 최대 수 (3.2초 동안 생성 최대 32회 x 양쪽)


@njit(cache=True)
def _run_ext(left0, top0, vx0, vy0, grounded0, inv0, bl_in, bt, bv, balive_in, g_in, gn_in, stars, salive_in,
             act, solid, spawn_rel, spawn_offset, out_state, out_bl, out_balive, out_g, out_gn, out_salive):
    """_run과 같은 규칙 + 가상 킬러. 생성 시각(spawn_rel - spawn_offset 프레임)이 되면 그 순간 마리오 top 높이에
    양쪽에서 가상 킬러(프레임당 4px)를 추가. 가상 킬러는 사고로 치지 않고 첫 접촉 프레임만 기록.
    반환: (실제 킬러/낙하 사고 프레임(-1 없음), 가상 킬러 첫 접촉 프레임(-1 없음)). 사고가 없으면 끝 상태를 out_*에 기록."""
    left, top, vx, vy, grounded, inv = left0, top0, vx0, vy0, grounded0, inv0
    dt = 1.0 / 60.0
    nb = bl_in.shape[0]
    bl = bl_in.copy()
    balive = balive_in.copy()
    gl = np.empty(GMAX)
    gt = np.empty(GMAX)
    gs = np.empty(GMAX)
    galive = np.zeros(GMAX, dtype=np.bool_)
    gn = gn_in
    for i in range(gn):
        gl[i], gt[i], gs[i] = g_in[i, 0], g_in[i, 1], g_in[i, 2]
        galive[i] = True
    ns = stars.shape[0]
    salive = salive_in.copy()
    nsp = spawn_rel.shape[0]
    sp = 0
    while sp < nsp and spawn_rel[sp] <= spawn_offset:
        sp += 1
    ghost_contact = -1
    n_steps = act.shape[0]
    frame = 0
    for s in range(n_steps):
        fx_in = act[s, 0] * 100.0
        fy_in = -act[s, 1] * 1000.0
        for _ in range(4):
            frame += 1
            if grounded:
                fx = max(-100.0, min(100.0, fx_in))
            else:
                fx = max(-75.0, min(75.0, fx_in))
            vx += fx
            if grounded and fy_in < 0:
                vy = max(-1000.0, min(0.0, fy_in))
                grounded = False
            # ---- 킬러 생성 (env: apply_force 다음, tick 이전, 그 순간 마리오 top 기준) ----
            while sp < nsp and spawn_rel[sp] == frame + spawn_offset:
                if gn + 2 <= GMAX:
                    gl[gn], gt[gn], gs[gn] = -48.0, max(0.0, top), 4.0
                    galive[gn] = True
                    gn += 1
                    gl[gn], gt[gn], gs[gn] = 1296.0, max(0.0, top), -4.0
                    galive[gn] = True
                    gn += 1
                sp += 1
            c0 = max(0, int(left) // 48 - 2)
            c1 = min(26, int(left + 48) // 48 + 2)
            r0 = max(0, int(top) // 48 - 2)
            r1 = min(17, int(top + 48) // 48 + 2)
            if grounded:
                vx = max(-750.0, min(750.0, vx))
            else:
                vx = max(-500.0, min(500.0, vx))
            vy += 2000.0 * dt
            if abs(vx) < 10.0:
                vx = 0.0
            left += float(int(vx * dt))
            if left < 0:
                left = 0.0
            if left + 48 > 1296:
                left = 1296.0 - 48
            vx *= 0.90
            ca, cb = max(c0, int(left) // 48), min(c1, int(left + 47) // 48)
            ra, rb = max(r0, int(top) // 48), min(r1, int(top + 47) // 48)
            for gx in range(ca, cb + 1):
                for gy in range(ra, rb + 1):
                    if solid[gy, gx]:
                        tl, tt = gx * 48.0, gy * 48.0
                        if _overlap(left, top, left + 48, top + 48, tl, tt, tl + 48, tt + 48):
                            if vx > 0:
                                left = tl - 48
                            elif vx < 0:
                                left = tl + 48
            top += float(int(vy * dt))
            grounded = False
            ca, cb = max(c0, int(left) // 48), min(c1, int(left + 47) // 48)
            ra, rb = max(r0, int(top) // 48), min(r1, int(top + 47) // 48)
            for gx in range(ca, cb + 1):
                for gy in range(ra, rb + 1):
                    if solid[gy, gx]:
                        tl, tt = gx * 48.0, gy * 48.0
                        if _overlap(left, top, left + 48, top + 48, tl, tt, tl + 48, tt + 48):
                            if vy > 0:
                                top = tt - 48
                                vy = 0.0
                                grounded = True
                            elif vy < 0:
                                top = tt + 48
                                vy = 0.0
            if not grounded and vy >= 0:
                sup = False
                gy_below = int(top + 48) // 48
                for gx in range(max(c0, int(left) // 48), min(c1, int(left + 47) // 48) + 1):
                    for gy in range(max(r0, gy_below), min(r1, gy_below) + 1):
                        if solid[gy, gx]:
                            tl, tt = gx * 48.0, gy * 48.0
                            if top + 48 == tt and left + 48 > tl and left < tl + 48:
                                sup = True
                grounded = sup
                if sup:
                    vy = 0.0
            if inv > 0:
                inv -= 1
            if top + 48 > 864:
                return frame, ghost_contact
            for j in range(nb):
                if balive[j]:
                    bl[j] += float(int(bv[j] * dt))
                    if (bv[j] > 0 and bl[j] > 1296) or (bv[j] < 0 and bl[j] + 48 < 0):
                        balive[j] = False
            for i in range(gn):
                if galive[i]:
                    gl[i] += gs[i]
                    if (gs[i] > 0 and gl[i] > 1296) or (gs[i] < 0 and gl[i] + 48 < 0):
                        galive[i] = False
            for k in range(ns):
                if salive[k] and _overlap(left, top, left + 48, top + 48,
                                          stars[k, 0], stars[k, 1], stars[k, 2], stars[k, 3]):
                    inv += 600
                    salive[k] = False
                    break
            for j in range(nb):
                if balive[j] and _overlap(left, top, left + 48, top + 48, bl[j], bt[j], bl[j] + 48, bt[j] + 48):
                    if vy > 0 and top + 48 < bt[j] + 24:
                        vy = -500.0
                        balive[j] = False
                    elif inv > 0:
                        balive[j] = False
                    else:
                        return frame, ghost_contact
            if ghost_contact < 0 and inv <= 0:
                for i in range(gn):
                    if galive[i] and _overlap(left, top, left + 48, top + 48, gl[i], gt[i], gl[i] + 48, gt[i] + 48):
                        ghost_contact = frame
                        break
    out_state[0], out_state[1], out_state[2], out_state[3] = left, top, vx, vy
    out_state[4] = 1.0 if grounded else 0.0
    out_state[5] = inv
    for j in range(nb):
        out_bl[j] = bl[j]
        out_balive[j] = balive[j]
    k = 0
    for i in range(gn):
        if galive[i]:
            out_g[k, 0], out_g[k, 1], out_g[k, 2] = gl[i], gt[i], gs[i]
            k += 1
    out_gn[0] = k
    for j in range(ns):
        out_salive[j] = salive[j]
    return -1, ghost_contact


@njit(cache=True)
def viability(m5, bul, stars, inv, P1, P2, gstart, glen, K, spawn_rel, solid):
    """1단계 계획 P1(1.6초) 실행 후 끝 상태에서 2단계 계획 P2(1.6초)를 시험.
    반환: 1단계 사고 프레임(P1개), 묶음별 점수(실제 킬러), 묶음별 점수(가상 킬러 포함)
    점수: 1단계 사고 시 0.5 x t1, 1단계 안전 시 0.5 + 0.5 x (2단계 최선 t2), t = 사고 시각/1.6초 (최대 1)"""
    n1 = P1.shape[0]
    nb = bul.shape[0]
    ns = stars.shape[0]
    bl0 = np.empty(nb)
    bt = np.empty(nb)
    bv = np.empty(nb)
    for j in range(nb):
        bl0[j], bt[j], bv[j] = bul[j, 0], bul[j, 1], bul[j, 4]
    ba0 = np.ones(nb, dtype=np.bool_)
    sa0 = np.ones(ns, dtype=np.bool_)
    g0 = np.zeros((GMAX, 3))
    first1 = np.full(n1, -1, dtype=np.int64)
    gc1 = np.full(n1, -1, dtype=np.int64)
    st_out = np.zeros((n1, 6))
    bl_out = np.zeros((n1, nb))
    ba_out = np.zeros((n1, nb), dtype=np.bool_)
    g_out = np.zeros((n1, GMAX, 3))
    gn_out = np.zeros((n1, 1), dtype=np.int64)
    sa_out = np.zeros((n1, ns), dtype=np.bool_)
    for p in range(n1):
        c, gc = _run_ext(m5[0], m5[1], m5[2], m5[3], m5[4] > 0.5, inv, bl0, bt, bv, ba0, g0, 0, stars, sa0,
                         P1[p], solid, spawn_rel, 0, st_out[p], bl_out[p], ba_out[p], g_out[p], gn_out[p], sa_out[p])
        first1[p] = c
        gc1[p] = gc
    ng = gstart.shape[0]
    sk = np.zeros(ng)
    sg = np.zeros(ng)
    t_state = np.zeros(6)
    t_bl = np.zeros(nb)
    t_ba = np.zeros(nb, dtype=np.bool_)
    t_g = np.zeros((GMAX, 3))
    t_gn = np.zeros(1, dtype=np.int64)
    t_sa = np.zeros(ns, dtype=np.bool_)
    for g in range(ng):
        best_k = 0.0
        best_g = 0.0
        ex = 0
        for q in range(glen[g]):
            p = gstart[g] + q
            if first1[p] >= 0:
                v = 0.5 * min(first1[p] / 96.0, 1.0)
                best_k = max(best_k, v)
                best_g = max(best_g, v)
                continue
            if ex >= K:
                best_k = max(best_k, 0.5)
                continue
            ex += 1
            bk = 0.0
            bg = 0.0
            for r in range(P2.shape[0]):
                c2, gc2 = _run_ext(st_out[p, 0], st_out[p, 1], st_out[p, 2], st_out[p, 3], st_out[p, 4] > 0.5,
                                   int(st_out[p, 5]), bl_out[p], bt, bv, ba_out[p], g_out[p], gn_out[p, 0], stars,
                                   sa_out[p], P2[r], solid, spawn_rel, 96, t_state, t_bl, t_ba, t_g, t_gn, t_sa)
                tk = 1.0 if c2 < 0 else c2 / 96.0
                tg = tk
                if gc2 >= 0 and gc2 / 96.0 < tg:
                    tg = gc2 / 96.0
                bk = max(bk, tk)
                bg = max(bg, tg)
                if bk >= 1.0 and bg >= 1.0:
                    break
            vk = 0.5 + 0.5 * bk
            vg = 0.5 + 0.5 * bg
            if gc1[p] >= 0:                          # 1단계 안에서 이미 가상 킬러와 접촉
                vg = min(vg, 0.5 * gc1[p] / 96.0)
            best_k = max(best_k, vk)
            best_g = max(best_g, vg)
        sk[g] = best_k
        sg[g] = best_g
    return first1, sk, sg


# 2단계 계획 13개: 가만히, 즉시 점프 3가지, 기다렸다 점프 3가지, 좌우 이동 4가지, 이동+점프 2가지
L2_PLANS = [(0.0, 0, 0, 0.0), (0.0, 0, 0, 0.5), (0.0, 0, 0, 0.75), (0.0, 0, 0, 1.0),
            (0.0, 0, 4, 0.6), (0.0, 0, 8, 0.6), (0.0, 0, 8, 1.0),
            (1.0, 3, 0, 0.0), (-1.0, 3, 0, 0.0), (1.0, 6, 0, 0.0), (-1.0, 6, 0, 0.0),
            (1.0, 3, 0, 0.75), (-1.0, 3, 0, 0.75)]
L2_ACTIONS = np.zeros((len(L2_PLANS), SIM_STEPS, 2), dtype=np.float64)
for _p, (_mf, _nm, _d, _f) in enumerate(L2_PLANS):
    L2_ACTIONS[_p, :int(_nm), 0] = _mf
    if _f > 0:
        L2_ACTIONS[_p, int(_d), 1] = _f
# 묶음 15개 (v5 특징과 같은 순서): 즉시 5, 기다림 6, 이동 4
GSTART = np.array([0, 1, 2, 3, 4] + [1 + di * 4 for di in range(1, 7)] + [mi * _NJ for mi in range(1, 5)], dtype=np.int64)
GLEN = np.array([1] * 5 + [4] * 6 + [_NJ] * 4, dtype=np.int64)
EXPAND_K = 3


def spawn_rel_frames(frame, horizon=2 * SIM_STEPS * 4):
    """현재 프레임 이후 horizon 프레임 안의 킬러 생성 시각 (상대 프레임)"""
    i0 = np.searchsorted(SPAWN_FRAMES, frame, side='right')
    i1 = np.searchsorted(SPAWN_FRAMES, frame + horizon, side='right')
    return (SPAWN_FRAMES[i0:i1] - frame).astype(np.int64)


def plan_features_v10(m, bb, inv_frames, stars, vy, spawn_rel):
    """반환: v9와 같은 1단계 특징 15, 1단계 최선값, 2단계 점수(실제 킬러) 15, 2단계 점수(가상 킬러 포함) 15, 2단계 최선값"""
    mario = np.array([m[0], m[1], m[4], vy, 0.0 if m[7] > 0.5 else 1.0], dtype=np.float64)
    first1, sk, sg = viability(mario, np.ascontiguousarray(bb, np.float64), np.ascontiguousarray(stars, np.float64),
                               int(inv_frames), PLAN_ACTIONS, L2_ACTIONS, GSTART, GLEN, EXPAND_K,
                               np.ascontiguousarray(spawn_rel, np.int64), SOLID)
    horizon = SIM_STEPS * DT
    first = np.where(first1 > 0, first1 / 60.0, np.inf)
    t = np.minimum(first, horizon) / horizon
    feat = np.concatenate([t[NOW_IDX], [t[i].max() for i in WAIT_IDX],
                           [t[i].max() for i in MOVE_IDX]]).astype(np.float32)
    return feat, float(t.max()), sk.astype(np.float32), sg.astype(np.float32), float(sk.max())



# ===================== v12: 첫 행동 조합별 생존 점수 =====================
@njit(cache=True)
def teacher_table(m5, bul, stars, inv, P1, P2, spawn_rel, solid):
    """계획 P1 각각의 점수 (실제 킬러 기준 sk, 가상 킬러 포함 sg). 1단계 안전한 계획은 모두 2단계로 확장"""
    n1 = P1.shape[0]
    nb = bul.shape[0]
    ns = stars.shape[0]
    bl0 = np.empty(nb)
    bt = np.empty(nb)
    bv = np.empty(nb)
    for j in range(nb):
        bl0[j], bt[j], bv[j] = bul[j, 0], bul[j, 1], bul[j, 4]
    ba0 = np.ones(nb, dtype=np.bool_)
    sa0 = np.ones(ns, dtype=np.bool_)
    g0 = np.zeros((GMAX, 3))
    sk = np.zeros(n1)
    sg = np.zeros(n1)
    o_st = np.zeros(6)
    o_bl = np.zeros(nb)
    o_ba = np.zeros(nb, dtype=np.bool_)
    o_g = np.zeros((GMAX, 3))
    o_gn = np.zeros(1, dtype=np.int64)
    o_sa = np.zeros(ns, dtype=np.bool_)
    t_st = np.zeros(6)
    t_bl = np.zeros(nb)
    t_ba = np.zeros(nb, dtype=np.bool_)
    t_g = np.zeros((GMAX, 3))
    t_gn = np.zeros(1, dtype=np.int64)
    t_sa = np.zeros(ns, dtype=np.bool_)
    for p in range(n1):
        c, gc = _run_ext(m5[0], m5[1], m5[2], m5[3], m5[4] > 0.5, inv, bl0, bt, bv, ba0, g0, 0, stars, sa0,
                         P1[p], solid, spawn_rel, 0, o_st, o_bl, o_ba, o_g, o_gn, o_sa)
        if c >= 0:
            v = 0.5 * min(c / 96.0, 1.0)
            sk[p] = v
            sg[p] = v
            continue
        bk = 0.0
        bg = 0.0
        for r in range(P2.shape[0]):
            c2, gc2 = _run_ext(o_st[0], o_st[1], o_st[2], o_st[3], o_st[4] > 0.5, int(o_st[5]), o_bl, bt, bv,
                               o_ba, o_g, o_gn[0], stars, o_sa, P2[r], solid, spawn_rel, 96,
                               t_st, t_bl, t_ba, t_g, t_gn, t_sa)
            tk = 1.0 if c2 < 0 else c2 / 96.0
            tg = tk
            if gc2 >= 0 and gc2 / 96.0 < tg:
                tg = gc2 / 96.0
            bk = max(bk, tk)
            bg = max(bg, tg)
            if bk >= 1.0 and bg >= 1.0:
                break
        sk[p] = 0.5 + 0.5 * bk
        sg[p] = 0.5 + 0.5 * bg
        if gc >= 0:
            sg[p] = min(sg[p], 0.5 * gc / 96.0)
    return sk, sg


# 계획마다 첫 행동의 (점프 단계 번호, 이동 방향) -> 조합 번호 = 점프 x 3 + (이동 + 1)
PLAN_J = np.array([int(np.argmin(np.abs(JUMP_LEVELS - PLAN_ACTIONS[p, 0, 1]))) for p in range(PLAN_ACTIONS.shape[0])])
PLAN_MV = np.sign(PLAN_ACTIONS[:, 0, 0]).astype(np.int64)
PLAN_PAIR = PLAN_J * 3 + (PLAN_MV + 1)
N_PAIR = N_JUMP * 3
# 공정한 첫 행동 평가: 조합 15가지 x 공통 이어가기 13가지 (첫 스텝 = 조합, 나머지 23스텝 = 이어가기 계획)
PAIR_FIRST = np.array([(float(mv), float(JUMP_LEVELS[j])) for j in range(N_JUMP) for mv in (-1, 0, 1)])
N_CONT = L2_ACTIONS.shape[0]
PAIR_PLANS = np.zeros((N_PAIR * N_CONT, SIM_STEPS, 2), dtype=np.float64)
for _k in range(N_PAIR):
    for _c in range(N_CONT):
        PAIR_PLANS[_k * N_CONT + _c, 0] = PAIR_FIRST[_k]
        PAIR_PLANS[_k * N_CONT + _c, 1:] = L2_ACTIONS[_c, :SIM_STEPS - 1]


def pair_scores(m, bb, inv_frames, st, vy, spawn_rel):
    """첫 행동 조합 15가지(점프 j x 이동 왼/정지/오른, 번호 = j x 3 + 이동 + 1)의 (실제 킬러 점수, 가상 킬러 포함 점수)"""
    mario = np.array([m[0], m[1], m[4], vy, 0.0 if m[7] > 0.5 else 1.0], dtype=np.float64)
    sk, sg = teacher_table(mario, np.ascontiguousarray(bb, np.float64).reshape(-1, 5),
                           np.ascontiguousarray(st, np.float64).reshape(-1, 4), int(inv_frames),
                           PAIR_PLANS, L2_ACTIONS, np.ascontiguousarray(spawn_rel, np.int64), SOLID)
    sk = sk.reshape(N_PAIR, N_CONT)
    sg = sg.reshape(N_PAIR, N_CONT)
    bc = (sk * 1000.0 + sg).argmax(1)                              # 조합마다 최선 이어가기
    r = np.arange(N_PAIR)
    return sk[r, bc].astype(np.float32), sg[r, bc].astype(np.float32)



# ===================== v15: 포화 구간 집중 학습 =====================
def check_sim(n=500, seed=0):
    """numba 동작과 2단계 예측 속도 확인 (포화 구간처럼 킬러가 많은 상황 포함)"""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        air = rng.random() < 0.5
        h = float(rng.integers(0, 200)) if air else 0.0
        left = float(rng.integers(528, 720))
        m = np.array([left, 720 - h, left + 48, 768 - h, rng.uniform(-500, 500),
                      rng.uniform(-900, 800) if air else 0, 0, float(air)], np.float32)
        nb = rng.integers(0, 30)
        bb = np.zeros((nb, 5), np.float32)
        for j in range(nb):
            top = float(rng.integers(500, 720))
            x0 = float(rng.integers(-48, 1296))
            bb[j] = [x0, top, x0 + 48, top + 48, rng.choice([-1, 1]) * rng.uniform(200, 300)]
        cases.append((m, bb, int(rng.integers(0, 10800))))
    st = np.zeros((0, 4))
    m0, bb0, f0 = cases[0]
    plan_features_v10(m0, bb0, 0, st, float(m0[5]), spawn_rel_frames(f0))      # 첫 호출 컴파일
    t0 = time.perf_counter()
    diff = 0
    for m, bb, f in cases:
        feat, best, sk, sg, vb = plan_features_v10(m, bb, 0, st, float(m[5]), spawn_rel_frames(f))
    el = (time.perf_counter() - t0) / n * 1000
    # 1단계 특징이 v9 계산과 같은지
    for m, bb, f in cases[:100]:
        a_ = plan_features_v10(m, bb, 0, st, float(m[5]), spawn_rel_frames(f))[0]
        b_ = plan_features(m, bb, 0, st, float(m[5]))[0]
        diff += not np.allclose(a_, b_)
    t1 = time.perf_counter()
    for m, bb, f in cases[:200]:
        pair_scores(m, bb, 0, st, float(m[5]), spawn_rel_frames(f))
    el_p = (time.perf_counter() - t1) / 200 * 1000
    print(f"v12 조합 점수 계산 {el_p:.3f} ms/회 (특징 계산 1회당 추가)")
    el = el + el_p
    print(f"numba 사용: {HAS_NUMBA} | 1단계 특징 v9와 불일치 {diff}/100 | 2단계 예측 + 조합 점수 {el:.3f} ms/회 | "
          f"환경 16개 기준 예상 수집 속도 약 {16000 / (el * 16 * 1.3 + 2.0):,.0f} steps/s (신경망 업데이트 제외)")


class FeatureExtractor:
    """관측 딕셔너리 → 100차원 상태 + shaping용 위험도. 무적 경과 시간을 내부에서 셈."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.inv_steps = 0
        self.last_time = -1.0
        self.frame = 0
        self.vy_tracker = VyTracker()

    def __call__(self, obs: Dict[str, np.ndarray], info: Dict = None):
        if info is not None:
            t = float(info.get('time_elapsed', 0) or 0)
            if t == 0 or t < self.last_time:
                self.reset()
            self.last_time = t
            self.frame = int(round(t * 60))

        m = np.asarray(obs['mario'], dtype=np.float32)
        cx, cy = (m[0] + m[2]) / 2, (m[1] + m[3]) / 2
        vx, vy, inv, air = m[4], m[5], m[6] > 0.5, m[7] > 0.5
        height = max(0.0, GROUND - m[3])
        self.inv_steps = self.inv_steps + 1 if inv else 0

        # 손 뗐을 때 멈출 위치 (등비급수), 공중이면 착지 위치 (수평 속도 유지 가정)
        stop_x = cx + vx * DT * DECAY / (1 - DECAY)
        if air:
            t_land = (-vy + math.sqrt(max(vy * vy + 2 * GRAVITY * height, 0.0))) / GRAVITY
            land_x = cx + vx * t_land
        else:
            t_land, land_x = 0.0, cx
        ref_x = land_x if air else stop_x
        edge_danger = float(np.clip((abs(ref_x - PLAT_C) - (PLAT_HALF - EDGE_WARN)) / EDGE_WARN, 0, 1))
        if m[3] > GROUND + 2:            # 이미 지면 아래로 떨어지는 중
            edge_danger = 1.0

        mario_f = np.array([
            (cx - PLAT_C) / PLAT_HALF,
            (cx - (PLAT_C - PLAT_HALF)) / PLAT_HALF,
            ((PLAT_C + PLAT_HALF) - cx) / PLAT_HALF,
            vx / VX_SCALE,
            vy / JUMP_V,
            height / MAX_JUMP_H,
            float(air),
            float(inv),
            min(self.inv_steps / INV_SCALE, 1.0),
            np.clip((stop_x - PLAT_C) / PLAT_HALF, -3, 3),
            float(abs(stop_x - PLAT_C) > PLAT_HALF),
            np.clip((land_x - PLAT_C) / PLAT_HALF, -3, 3),
            float(air and abs(land_x - PLAT_C) > PLAT_HALF),
            min(t_land, 2.0) / 2.0,
        ], dtype=np.float32)

        # ---------- 킬러 ----------
        bullet_f = np.zeros((K_BULLETS, BULLET_FEAT), dtype=np.float32)
        bullet_danger = 0.0
        bb = np.asarray(obs['bullet_bills'], dtype=np.float32)
        bb = bb[np.abs(bb).sum(1) > 0]
        inv_frames = max(0, 600 - 4 * (self.inv_steps - 1)) if inv else 0
        st_now = np.asarray(obs['stars'], dtype=np.float32)
        st_now = st_now[np.abs(st_now).sum(1) > 0]
        vy64 = self.vy_tracker.update(m) if info is not None else None
        spawn_rel = spawn_rel_frames(self.frame)
        opt_f, plan_best, via_k, via_g, via_best = plan_features_v10(m, bb, inv_frames, st_now,
                                                                     vy64 if vy64 is not None else float(m[5]), spawn_rel)
        spawn_f = spawn_features(self.frame, self.last_time if self.last_time > 0 else 0.0)
        if len(bb):
            top, bot, bvx = bb[:, 1], bb[:, 3], bb[:, 4]
            bcx, bcy = (bb[:, 0] + bb[:, 2]) / 2, (top + bot) / 2
            dx = bcx - cx
            closing = -np.sign(dx) * (bvx - vx)              # 양수면 가까워지는 중
            gap = np.abs(dx) - SIZE
            ttc = np.where(gap <= 0, 0.0,
                           np.where(closing > 1e-3, gap / np.maximum(closing, 1e-3), TTC_MAX))
            ttc = np.minimum(ttc, TTC_MAX)
            v_overlap = np.abs(bcy - cy) < SIZE
            if not inv and v_overlap.any():
                bullet_danger = float(np.clip(1 - ttc[v_overlap] / 0.8, 0, 1).max())
            order = np.argsort(ttc + np.abs(dx) / 1e4)[:K_BULLETS]
            feats = np.stack([
                np.ones_like(dx),
                np.clip(dx / 600.0, -2, 2),
                (bcy - cy) / MAX_JUMP_H,
                bvx / BB_VX_SCALE,
                (closing > 0).astype(np.float32),
                ttc / TTC_MAX,
                (top > GROUND_LANE_TOP).astype(np.float32),
                (top >= STOMPABLE_TOP).astype(np.float32),
                v_overlap.astype(np.float32),
                (top - m[3]) / MAX_JUMP_H,
            ], axis=1)[order]
            bullet_f[:len(feats)] = feats

        # ---------- 별 ----------
        star_f = np.zeros((N_STARS, STAR_FEAT), dtype=np.float32)
        st = np.asarray(obs['stars'], dtype=np.float32)[:N_STARS]
        for k, row in enumerate(st):
            if np.abs(row).sum() > 0:
                scx, scy = (row[0] + row[2]) / 2, (row[1] + row[3]) / 2
                star_f[k] = [1.0, (scx - cx) / 600.0, (scy - cy) / 400.0]

        psk, psg = pair_scores(m, bb, inv_frames, st_now, vy64 if vy64 is not None else float(m[5]), spawn_rel)
        self.last_pair = (psk, psg)                                     # 분석용 (정책이 점수를 어떻게 쓰는지)
        s = np.concatenate([mario_f, opt_f, spawn_f, bullet_f.ravel(), star_f.ravel(), via_k, via_g,
                            psk, psg]).astype(np.float32)
        # v9: 보상용 위험도는 '무적 아님, 별 없음'으로 가정해 계산 (별을 먹는 것 자체가 보상이 되지 않도록)
        #     무적 중이거나 별이 가까울 때만 다시 계산하고, 그 외에는 상태 특징과 같은 값
        near_star = len(st_now) > 0 and np.min(np.abs((st_now[:, 0] + st_now[:, 2]) / 2 - (m[0] + m[2]) / 2)
                                               + np.abs((st_now[:, 1] + st_now[:, 3]) / 2 - (m[1] + m[3]) / 2)) < 450
        if inv or near_star:
            via_best = plan_features_v10(m, bb, 0, np.zeros((0, 4), np.float32),
                                         vy64 if vy64 is not None else float(m[5]), spawn_rel)[4]
        bullet_danger = 1.0 - via_best        # v10: 3.2초 동안 계속 살 수 있는 길이 있어야 위험 0
        return s, edge_danger, bullet_danger


def potential(edge_danger: float, bullet_danger: float) -> float:
    # v7: 발판 끝 위험(가운데 땅만 안전하다는 가정) 제거, 계획 기반 위험도(정확한 시뮬레이션, 낙하 포함)만 사용
    return -(BULLET_W * bullet_danger)




# ================= 신경망 =================
def layer(inp, out, gain=math.sqrt(2)):
    lin = nn.Linear(inp, out)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


class ActorCritic(nn.Module):
    """v14: 정책(15가지 조합 범주형)과 가치 함수. 둘은 몸통을 공유하지 않음.
    학습 코드와의 호환을 위해 act/evaluate 는 (조합 번호, 0, log 확률) 형태를 유지"""

    def __init__(self, obs_dim=OBS_DIM, hidden=256):
        super().__init__()
        self.actor_body = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh())
        self.pair_head = layer(hidden, N_ACT, gain=0.01)
        self.critic = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(),
                                    layer(hidden, 1, gain=1.0))

    def logits(self, o):
        return self.pair_head(self.actor_body(o))

    def dist(self, o):
        return Categorical(logits=self.logits(o))

    def value(self, o):
        return self.critic(o).squeeze(-1)

    def act(self, o, deterministic=False):
        d = self.dist(o)
        k = d.probs.argmax(-1) if deterministic else d.sample()
        return k, torch.zeros_like(k, dtype=torch.float32), d.log_prob(k)

    def evaluate(self, o, k, _m=None):
        d = self.dist(o)
        return d.log_prob(k), d.entropy(), torch.zeros(len(k)), self.value(o)


# ================= 제출용 에이전트 =================
class BulletBillAgent(kym.Agent):
    def __init__(self, model: ActorCritic = None, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.model = model if model is not None else ActorCritic(hidden=hidden)
        self.fe = FeatureExtractor()
        self.deterministic = True        # False면 학습 때처럼 확률적으로 행동 선택

    def act(self, observation: Any, info: Dict):
        s, _, _ = self.fe(observation, info)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            k, _, _ = self.model.act(torch.as_tensor(s, device=device).unsqueeze(0), deterministic=self.deterministic)
        return pair_to_env(k.item())

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = ActorCritic(obs_dim=ckpt['obs_dim'], hidden=ckpt['hidden'])
        model.load_state_dict(ckpt['model'])
        model.eval()
        return cls(model, hidden=ckpt['hidden'])

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'model': {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    'obs_dim': OBS_DIM, 'hidden': self.hidden}, path)


# ================= 사망 원인 / 평가 =================
def death_cause(obs, info):
    """사망 원인: water / bullet_{ground|air}_{stand|jump} / cleared
    마리오 상자와 겹치는(여유 6px) 킬러를 충돌 상대로 봄. 겹치는 게 없으면 가장 가까운 킬러."""
    if info.get('game_state') == 'cleared':
        return 'cleared'
    m = obs['mario']
    if m[3] > GROUND + 30:
        return 'water'
    bb = obs['bullet_bills']
    bb = bb[np.abs(bb).sum(1) > 0]
    if not len(bb):
        return 'bullet_unknown'
    mg = 6.0
    ov = (bb[:, 0] < m[2] + mg) & (bb[:, 2] > m[0] - mg) & (bb[:, 1] < m[3] + mg) & (bb[:, 3] > m[1] - mg)
    cand = bb[ov] if ov.any() else bb
    cx, cy = (m[0] + m[2]) / 2, (m[1] + m[3]) / 2
    dist = np.hypot((cand[:, 0] + cand[:, 2]) / 2 - cx, (cand[:, 1] + cand[:, 3]) / 2 - cy)
    b = cand[dist.argmin()]
    lane = 'ground' if b[1] > GROUND_LANE_TOP else 'air'
    pose = 'jump' if m[7] > 0.5 else 'stand'
    return f'bullet_{lane}_{pose}' + ('' if ov.any() else '_near')


CAUSE_KO = {'water': '물', 'bullet_ground_stand': '지면 킬러(서 있다가)',
            'bullet_ground_jump': '지면 킬러(점프 중)', 'bullet_air_jump': '공중 킬러(점프 중)',
            'bullet_air_stand': '공중 킬러(서 있다가)', 'bullet_unknown': '킬러(불명)', 'cleared': '클리어'}


def cause_summary(causes):
    from collections import Counter
    c = Counter(causes)
    return ', '.join(f"{CAUSE_KO.get(k.replace('_near', ''), k)}{'(근접추정)' if k.endswith('_near') else ''} {v}" for k, v in c.most_common())


def evaluate(agent: BulletBillAgent, seeds):
    env = gym.make(id=ENV_ID, render_mode='none', bgm=False)
    times, causes = [], []
    for sd in seeds:
        obs, info = env.reset(seed=int(sd))
        agent.fe.reset()
        while True:
            obs, _, term, trunc, info = env.step(agent.act(obs, info))
            if term or trunc:
                break
        times.append(float(info['time_elapsed']))
        causes.append(death_cause(obs, info))
    env.close()
    return np.array(times), causes


def compute_gae(R, V, D, last_v, gamma, lam):
    """R, V, D: (T, N). D[t] = 1 이면 t번째 행동 직후 에피소드 종료."""
    T = len(R)
    adv = np.zeros_like(R)
    last = np.zeros(R.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_v if t == T - 1 else V[t + 1]
        nonterm = 1.0 - D[t]
        delta = R[t] + gamma * next_v * nonterm - V[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv



# ================= Standalone parallel PPO/DAgger =================
import json
import multiprocessing as mp
import queue
import traceback
from pathlib import Path
from torch.distributions import Categorical


def _atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def _cpu_weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _rollout_worker(wid, incoming, outgoing, seed):
    torch.set_num_threads(1)
    env = None
    try:
        import gymnasium as gym
        import kymnasium  # noqa: F401, environment registration
        env = gym.make(id=ENV_ID, render_mode='none', bgm=False)
        model = ActorCritic().cpu().eval()
        feature = FeatureExtractor()
        rng = np.random.default_rng(seed + 1000003 * wid)
        state = obs = info = None
        phi = 0.0
        while True:
            msg = incoming.get()
            if msg is None:
                break
            wave, weights, count = msg
            model.load_state_dict(weights, strict=True)
            rows = {k: [] for k in ('s', 'a', 'logp', 'v', 'r', 'done')}
            survived = []
            with torch.inference_mode():
                for _ in range(count):
                    if obs is None:
                        obs, info = env.reset(seed=int(rng.integers(1, 2147483000)))
                        feature.reset()
                        state, ed, bd = feature(obs, info)
                        phi = potential(ed, bd)
                    x = torch.from_numpy(state).unsqueeze(0)
                    action, _, logprob = model.act(x)
                    value = model.value(x)
                    idx = int(action.item())
                    env_action = pair_to_env(idx)
                    on_ground = obs['mario'][7] < 0.5
                    obs2, _, terminated, truncated, info2 = env.step(env_action)
                    state2, ed2, bd2 = feature(obs2, info2)
                    done = bool(terminated or truncated)
                    dead = terminated and info2.get('game_state') == 'game_over'
                    phi2 = 0.0 if done else potential(ed2, bd2)
                    reward = SURVIVE + SHAPE * (GAMMA * phi2 - phi)
                    if on_ground and env_action[1] > 0:
                        reward += JUMP_COST
                    if dead:
                        reward += DEATH
                    if obs2['mario'][6] > 0.5 and obs['mario'][6] < 0.5:
                        reward -= STAR_EARLY_PENALTY * max(0., 1. - float(info2['time_elapsed']) / STAR_FREE_TIME)
                    rows['s'].append(state)
                    rows['a'].append(idx)
                    rows['logp'].append(float(logprob.item()))
                    rows['v'].append(float(value.item()))
                    rows['r'].append(float(reward))
                    rows['done'].append(float(done))
                    if done:
                        survived.append(float(info2['time_elapsed']))
                        state = obs = info = None
                        phi = 0.
                    else:
                        state, obs, info, phi = state2, obs2, info2, phi2
                final_value = 0.0 if state is None else float(model.value(torch.from_numpy(state).unsqueeze(0)).item())
            out = {key: np.asarray(value, dtype=(np.int64 if key == 'a' else np.float32))
                   for key, value in rows.items()}
            out['last_value'] = final_value
            out['survived'] = survived
            outgoing.put((wid, wave, out, None))
    except BaseException:
        outgoing.put((wid, -1, None, traceback.format_exc()))
    finally:
        if env is not None:
            env.close()


def _gae(roll, gamma, lam):
    r, v, d = roll['r'], roll['v'], roll['done']
    advantage = np.empty(len(r), dtype=np.float32)
    carry = 0.
    for t in range(len(r) - 1, -1, -1):
        nextv = roll['last_value'] if t == len(r) - 1 else float(v[t + 1])
        mask = 1. - float(d[t])
        delta = float(r[t]) + gamma * nextv * mask - float(v[t])
        carry = delta + gamma * lam * mask * carry
        advantage[t] = carry
    return advantage, advantage + v


def _update_ppo(model, optimizer, rolls, args, device, imitation_coef, entropy_coef):
    data = {k: [] for k in ('s', 'a', 'logp', 'adv', 'ret')}
    for roll in rolls:
        adv, ret = _gae(roll, GAMMA, args.gae_lambda)
        for key, value in (('s', roll['s']), ('a', roll['a']), ('logp', roll['logp']),
                           ('adv', adv), ('ret', ret)):
            data[key].append(value)
    data = {key: torch.from_numpy(np.concatenate(vals)).to(device) for key, vals in data.items()}
    adv = data['adv']
    data['adv'] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    model.train()
    metrics = []
    for _ in range(args.epochs):
        for indices in torch.randperm(len(adv), device=device).split(args.minibatch):
            states = data['s'][indices]
            logits = model.logits(states)
            dist = Categorical(logits=logits)
            lp = dist.log_prob(data['a'][indices])
            ent = dist.entropy().mean()
            vpred = model.value(states)
            ratio = (lp - data['logp'][indices]).exp()
            adv_mb = data['adv'][indices]
            actor = torch.maximum(-adv_mb * ratio, -adv_mb * ratio.clamp(1. - args.clip, 1. + args.clip)).mean()
            value = 0.5 * (vpred - data['ret'][indices]).square().mean()
            scores = states[:, PAIR_OFF:PAIR_OFF + N_ACT]
            best = scores >= (scores.amax(dim=1, keepdim=True) - 1e-6)
            imitation = (torch.logsumexp(logits, dim=1) -
                         torch.logsumexp(logits.masked_fill(~best, -1e9), dim=1)).mean()
            loss = actor + args.vf_coef * value - entropy_coef * ent + imitation_coef * imitation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1.) - (lp - data['logp'][indices])).mean()
                metrics.append((float(actor), float(value), float(imitation), float(ent), float(approx_kl)))
    model.eval()
    means = np.mean(metrics, axis=0)
    return {key: round(float(value), 6) for key, value in zip(
        ('actor_loss', 'value_loss', 'imit_loss', 'entropy', 'approx_kl'), means)}


def train(total_steps=50_000_000, envs=16, batch_steps=256, device='auto', seed=42,
          lr=1e-4, epochs=4, minibatch=1024, gae_lambda=.95, clip=.2,
          vf_coef=.5, max_grad_norm=.5, ent0=.01, ent1=.003,
          imit0=1., imit1=.05, imit_steps=5_000_000,
          checkpoint_every=2_000_000, log_every=100,
          out='models/parallel50m_final.pt', checkpoint='models/parallel50m_resume.pt',
          resume=None):
    """Train from random initialization if resume is None; no midway evaluation."""
    args = argparse.Namespace(gae_lambda=gae_lambda, clip=clip, vf_coef=vf_coef,
                              max_grad_norm=max_grad_norm, epochs=epochs, minibatch=minibatch)
    if min(total_steps, envs, batch_steps, epochs, minibatch, checkpoint_every, log_every) <= 0:
        raise ValueError('step counts, number of workers, epochs, and intervals must be positive')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')
    if not HAS_NUMBA:
        raise RuntimeError('Numba is required for practical 50M training. Install numba.')
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    done = 0
    if resume:
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        if ck.get('trainer') != 'single_file_ppo_parallel_v1':
            raise ValueError('Resume requires a checkpoint from this exact parallel trainer')
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        for state in optimizer.state.values():
            for k, value in state.items():
                if torch.is_tensor(value):
                    state[k] = value.to(device)
        done = int(ck['total_steps'])
        if ck['target_steps'] != total_steps:
            raise ValueError('Pass the original TOTAL target steps when resuming')
    if done >= total_steps:
        raise ValueError('Checkpoint already reached requested target')
    if Path(out).resolve() == Path(checkpoint).resolve():
        raise ValueError('Final model and recovery checkpoint must have distinct paths')
    ctx = mp.get_context('spawn')
    inputs = [ctx.Queue(maxsize=2) for _ in range(envs)]
    outputs = ctx.Queue(maxsize=max(4, 2 * envs))
    workers = [ctx.Process(target=_rollout_worker, args=(i, inputs[i], outputs, seed), daemon=True)
               for i in range(envs)]
    for worker in workers:
        worker.start()
    batch = envs * batch_steps
    wave = 0
    last_ckpt = done
    t_start = time.perf_counter()
    print('START ' + json.dumps({'total_steps': total_steps, 'starting_steps': done,
                                  'batch': batch, 'workers': envs, 'device': str(device),
                                  'evaluation': 'disabled', 'environment': 'unmodified'}, ensure_ascii=False), flush=True)
    try:
        while done < total_steps:
            remain = total_steps - done
            counts = [max(0, min(batch_steps, remain - i * batch_steps)) for i in range(envs)]
            active = [i for i, count in enumerate(counts) if count > 0]
            weights = _cpu_weights(model)
            for i in active:
                inputs[i].put((wave, weights, counts[i]))
            collected = {}
            while len(collected) < len(active):
                try:
                    wid, received_wave, result, err = outputs.get(timeout=10)
                except queue.Empty:
                    if any(not workers[i].is_alive() for i in active):
                        raise RuntimeError('Rollout worker terminated unexpectedly')
                    continue
                if err:
                    raise RuntimeError(f'Worker {wid} failed:\n{err}')
                if wid not in active or received_wave != wave:
                    raise RuntimeError('Out-of-order rollout response')
                collected[wid] = result
            rolls = [collected[i] for i in active]
            fraction = done / max(total_steps, 1)
            coef_imit = imit0 + (imit1 - imit0) * min(1., done / imit_steps)
            coef_ent = ent0 + (ent1 - ent0) * fraction
            metrics = _update_ppo(model, optimizer, rolls, args, device, coef_imit, coef_ent)
            done += sum(len(r['a']) for r in rolls)
            wave += 1
            if wave == 1 or wave % log_every == 0 or done == total_steps:
                elapsed = time.perf_counter() - t_start
                finished = [x for roll in rolls for x in roll['survived']]
                print('UPDATE ' + json.dumps({'wave': wave, 'steps': done, 'target': total_steps,
                                              'steps_per_s': round((done - (0 if not resume else ck['total_steps'])) / max(elapsed, 1e-6), 1),
                                              'episodes_ended': len(finished),
                                              'mean_ended_s': round(float(np.mean(finished)), 3) if finished else None,
                                              'imit_coef': round(coef_imit, 5), 'ent_coef': round(coef_ent, 5), **metrics}, ensure_ascii=False), flush=True)
            if done - last_ckpt >= checkpoint_every or done == total_steps:
                _atomic_save({'trainer': 'single_file_ppo_parallel_v1', 'model': _cpu_weights(model),
                              'optimizer': optimizer.state_dict(), 'obs_dim': OBS_DIM, 'hidden': 256,
                              'target_steps': total_steps, 'total_steps': done, 'seed': seed}, checkpoint)
                last_ckpt = done
                print('CHECKPOINT ' + str(checkpoint), flush=True)
        _atomic_save({'model': _cpu_weights(model), 'obs_dim': OBS_DIM, 'hidden': 256,
                      'total_steps': done, 'trained': True}, out)
        print('FINAL ' + json.dumps({'model': str(out), 'total_steps': done}, ensure_ascii=False), flush=True)
    finally:
        for q in inputs:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for worker in workers:
            worker.join(timeout=1)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1)
        for q in inputs:
            q.close()
        outputs.close()


def main():
    parser = argparse.ArgumentParser(description='Single-file PPO+DAgger, 16-process parallel 50M training')
    parser.add_argument('--steps', type=int, default=50_000_000, help='TOTAL training environment transitions')
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--batch-steps', type=int, default=256)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--minibatch', type=int, default=1024)
    parser.add_argument('--ent0', type=float, default=.01)
    parser.add_argument('--ent1', type=float, default=.003)
    parser.add_argument('--imit0', type=float, default=1.)
    parser.add_argument('--imit1', type=float, default=.05)
    parser.add_argument('--imit-steps', type=int, default=5_000_000)
    parser.add_argument('--checkpoint-every', type=int, default=2_000_000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--out', default='models/parallel50m_final.pt')
    parser.add_argument('--checkpoint', default='models/parallel50m_resume.pt')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--model', default='models/parallel50m_final.pt')
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--demo', action='store_true')
    a = parser.parse_args()
    if a.demo:
        agent = BulletBillAgent.load(a.model)
        kym.evaluate(env_id=ENV_ID, agent=agent, render_mode='human', bgm=True, seed=a.seed)
    elif a.eval:
        agent = BulletBillAgent.load(a.model)
        times, causes = evaluate(agent, range(20000, 20000 + a.eval))
        print('EVAL ' + json.dumps({'n': a.eval, 'mean_s': round(float(times.mean()), 3),
                                   'median_s': round(float(np.median(times)), 3),
                                   'cleared': int((times >= 180).sum()), 'clear_rate': round(float(np.mean(times >= 180)), 4)}, ensure_ascii=False))
    else:
        train(total_steps=a.steps, envs=a.envs, batch_steps=a.batch_steps, device=a.device,
              seed=a.seed, lr=a.lr, epochs=a.epochs, minibatch=a.minibatch,
              ent0=a.ent0, ent1=a.ent1, imit0=a.imit0, imit1=a.imit1,
              imit_steps=a.imit_steps, checkpoint_every=a.checkpoint_every,
              log_every=a.log_every, out=a.out, checkpoint=a.checkpoint, resume=a.resume)


if __name__ == '__main__':
    mp.freeze_support()
    main()
