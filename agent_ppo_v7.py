"""
Bullet Bills - 하이브리드 행동 PPO 에이전트 (v7: 정확한 시뮬레이터 + 킬러 생성 타이밍)
KNU Advanced RL Competition Round 2

[실행]
  python agent_ppo_v7.py                  # 학습 (기본 1,000만 스텝, 환경 16개 동시)
  python agent_ppo_v7.py --steps 2000000  # 짧게 학습해서 v6와 비교
  python agent_ppo_v7.py --eval 200 --model models/bullet_bill_ppo_v7_best.pt
  python agent_ppo_v7.py --check-sim      # numba 동작, 계획 특징 계산 속도 확인
[결과]
  models/bullet_bill_ppo_v7_best.pt       평가 성능이 가장 좋았던 정책 (시연용)
  models/bullet_bill_ppo_v7_last.pt       마지막 정책
[v6 대비 변경]
  1) 계획 예측 시뮬레이터 교체: 환경 소스(env.py/objs.py/consts.py) 규칙을 프레임 단위로 옮긴 정확한 시뮬레이터
     - 60fps x 4프레임, 정수 좌표 이동, 블록 79개 충돌(위에 서기, 머리 박기, 옆면), 킬러 프레임당 정수 픽셀 이동
     - 밟기(낙하 중 + 마리오 bottom < 킬러 중심), 별(무적 10초), 낙하 사망(bottom > 864)
     - 계획 145개는 v5와 같은 구성(즉시/기다림/이동)을 스텝별 행동 배열로 표현
  2) 발판 끝 위험 보상 제거: Phi = -(계획 기반 위험도) 만 사용 (낙하는 정확한 시뮬레이션에 포함됨)
     -> 가운데 땅만 안전하다고 가정해 피라미드·벽돌 쪽 이동을 막던 보상 제거
  3) 킬러 생성 타이밍 특징 3개: 다음 두 번의 생성까지 남은 시간, 현재 생성 간격
     - 생성 규칙(env.py): 높이 = 마리오 top + U(-144, 144), 마리오 top보다 낮게는 안 나옴(절반은 정확히 마리오 높이)
       생성 시각 = 타이머 기반, 간격 1.5 * (0.1/1.5)^(t/105)초 (무작위 아님) -> 경과 시간으로 정확히 계산
     - 생성 순간 마리오가 48px 이상 떠 있으면 그 킬러는 서 있는 마리오 위로 지나감 -> 학습으로 활용 가능
  상태 118차원 = v1 특징 100 + 계획 15 + 생성 타이밍 3
[v5 대비 변경 (v6)]
  보상의 킬러 위험 항을 v1 방식(현재 높이에서 겹치는 킬러가 0.8초 안에 도착하는가)에서
  계획 기반 위험도로 교체: danger = 1 - (145개 계획 중 최선 계획의 첫 사고 시각 / 1.6초)
    -> 안전한 계획이 하나라도 있으면 위험 0, 안전한 길을 없애는 행동을 하면 그 순간 음의 보상
    -> 계획 시뮬레이션에 물이 포함되므로 물 위험도 반영
  Phi = -(발판 끝 위험 + 1.0 * 계획 위험), F = 0.3 * (gamma * Phi(s') - Phi(s))  (potential-based, 최적 정책 불변)
  근거: v5 긴 학습 사망 200판 분석에서 사망 1초 전 92%에 안전한 계획이 있었고(상태에도 그 정보가 있음),
        판단 실수 46%. 실수(안전한 길을 잃는 행동)와 벌(사망)이 0.5~1초 떨어져 있어 학습이 어려움.
        또 v1 위험도는 공중 킬러 사망 직전 점프의 27%를 칭찬하고 있었음.
[v4 대비 변경 (v5)]
  예측 특징을 5차원(즉시 점프 선택지) -> 15차원(계획)으로 확장 (105 -> 115차원)
    - 즉시 5 : 안 뜀, 0.5, 0.6, 0.75, 1.0으로 지금 점프 (v4와 동일)
    - 기다림 6: 2/4/6/8/10/12스텝 기다린 뒤 점프할 때 높이별 최선의 첫 사고 시각
    - 이동 4 : 오른쪽/왼쪽 최대 힘으로 3/6스텝 이동(이후 점프 계획 포함)할 때 최선의 첫 사고 시각
  근거: v4 사망 200판 분석에서 사망 1초 전 상태의 44%는 '기다렸다 점프'로, 13%는 '이동'으로
        피할 수 있었고 즉시 점프 선택지만으로는 이를 구분할 수 없었음 (피할 수 없던 죽음은 4.6%)
  이동 모델(측정): vx' = 0.656 * vx + 309.5 * 이동 힘
[v3 대비 변경 (v4)]
  상태에 점프 선택지(안 뜀, 0.5, 0.6, 0.75, 1.0)별 '첫 사고(킬러 충돌 또는 물) 예측 시각'을 추가 (100 -> 105차원)
  예측 시뮬레이션은 실제 환경 재현 결과 2,735건과 대조해 보정함 (지면 상태 정확도 95.2%):
    - 공중에서도 이동 입력이 없으면 수평 속도가 스텝당 0.656배로 감쇠
    - 착지/밟기 순간에 누른 점프는 바로 실행됨
    - 충돌 판정은 그림 크기(48px)와 거의 같음 (축소 2px)

[방법 요약]
  알고리즘 : PPO (Schulman et al., 2017), clipped objective, GAE(lambda=0.95), 환경 16개 x 256스텝 = 배치 4096
  행동     : 하이브리드 — 점프는 범주형 5개(안 뜀, 0.5, 0.6, 0.75, 1.0), 이동은 가우시안 연속값
             log pi(a|s) = log pi(점프|s) + log pi(이동|s)
  선택 근거: 점프는 '뛸지 말지'의 이산 결정이라 Q의 행동 미분을 쓰는 SAC로는 배우기 어려움.
             환경이 초당 1만 스텝 이상으로 빨라 경험이 싸므로, 재사용(off-policy)보다 대량 수집(on-policy)이 유리.
  상태·보상: v1의 100차원 + 계획 예측 15차원, 보상은 v1과 동일
  측정한 물리: 1스텝 1/15초, 발판(마리오 중심) 504~792, 손 뗐을 때 속도 0.656배/스텝,
               점프 높이 약 226*힘^2, 지면 킬러 top > 672
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
OBS_DIM = MARIO_FEAT + PLAN_FEAT + 3 + K_BULLETS * BULLET_FEAT + N_STARS * STAR_FEAT   # 118 (생성 타이밍 3)
ACT_DIM = 2

# ================= 보상 =================
SURVIVE, DEATH, JUMP_COST = 0.1, -10.0, -0.02
SHAPE, BULLET_W = 0.3, 1.0          # v6: 킬러 위험 = 계획 기반 위험도
GAMMA = 0.99

# ================= 행동 =================
JUMP_LEVELS = np.array([0.0, 0.5, 0.6, 0.75, 1.0], dtype=np.float32)   # 안 뜀, 57/81/127/226px
N_JUMP = len(JUMP_LEVELS)


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


SPAWN_FRAMES = spawn_frames()


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


def check_sim(n=2000, seed=0):
    """numba 동작과 계획 특징 계산 속도 확인 (시뮬레이터 정확도는 validate_exact.py로 검증)"""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        air = rng.random() < 0.5
        h = float(rng.integers(0, 200)) if air else 0.0
        left = float(rng.integers(528, 720))
        m = np.array([left, 720 - h, left + 48, 768 - h, rng.uniform(-500, 500),
                      rng.uniform(-900, 800) if air else 0, 0, float(air)], np.float32)
        nb = rng.integers(0, 10)
        bb = np.zeros((nb, 5), np.float32)
        for j in range(nb):
            top = 720.0 if rng.random() < 0.5 else float(rng.integers(576, 720))
            x0 = float(rng.integers(-48, 1296))
            bb[j] = [x0, top, x0 + 48, top + 48, rng.choice([-1, 1]) * rng.uniform(200, 300)]
        cases.append((m, bb))
    st = np.zeros((0, 4))
    plan_features(cases[0][0], cases[0][1], 0, st)            # 첫 호출 컴파일
    t0 = time.perf_counter()
    for m, bb in cases:
        plan_features(m, bb, 0, st)
    el = (time.perf_counter() - t0) / n * 1000
    print(f"numba 사용: {HAS_NUMBA} | 계획 특징(정확한 시뮬레이터, 계획 145개) {el:.3f} ms/회 | "
          f"환경 16개 기준 예상 수집 속도 약 {16000 / (el * 16 + 2.0):,.0f} steps/s (신경망 업데이트 시간 제외)")


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
        opt_f, plan_best = plan_features(m, bb, inv_frames, st_now, vy64)
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

        s = np.concatenate([mario_f, opt_f, spawn_f, bullet_f.ravel(), star_f.ravel()]).astype(np.float32)
        bullet_danger = 1.0 - plan_best      # v6: 최선 계획조차 사고가 나는 정도 (v1 겹침 방식 대체)
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
    """정책(점프 범주형 + 이동 가우시안)과 가치 함수. 둘은 몸통을 공유하지 않음."""

    def __init__(self, obs_dim=OBS_DIM, hidden=256):
        super().__init__()
        self.actor_body = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh())
        self.jump_head = layer(hidden, N_JUMP, gain=0.01)
        self.move_head = layer(hidden, 1, gain=0.01)
        self.move_log_std = nn.Parameter(torch.full((1,), -0.5))
        self.critic = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(),
                                    layer(hidden, 1, gain=1.0))

    def dists(self, o):
        h = self.actor_body(o)
        dj = Categorical(logits=self.jump_head(h))
        dm = Normal(self.move_head(h).squeeze(-1), self.move_log_std.exp())
        return dj, dm

    def value(self, o):
        return self.critic(o).squeeze(-1)

    def act(self, o, deterministic=False):
        dj, dm = self.dists(o)
        if deterministic:
            j, m = dj.probs.argmax(-1), dm.mean
        else:
            j, m = dj.sample(), dm.sample()
        return j, m, dj.log_prob(j) + dm.log_prob(m)

    def evaluate(self, o, j, m):
        dj, dm = self.dists(o)
        return dj.log_prob(j) + dm.log_prob(m), dj.entropy(), dm.entropy(), self.value(o)


# ================= 제출용 에이전트 =================
class BulletBillAgent(kym.Agent):
    def __init__(self, model: ActorCritic = None, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.model = model if model is not None else ActorCritic(hidden=hidden)
        self.fe = FeatureExtractor()

    def act(self, observation: Any, info: Dict):
        s, _, _ = self.fe(observation, info)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            j, m, _ = self.model.act(torch.as_tensor(s, device=device).unsqueeze(0), deterministic=True)
        return to_env_action(m.item(), j.item())

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        ckpt = torch.load(path, map_location='cpu')
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


# ================= 학습 =================
def train(total_steps=10_000_000, seed=42, tag='', n_envs=16, n_steps=256, epochs=4, minibatches=4,
          lr=3e-4, lam=0.95, clip=0.2, ent_jump=0.01, ent_move=0.001, vf_coef=0.5, max_grad=0.5,
          eval_interval=500_000, eval_episodes=30, log_every=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    suffix = f'_{tag}' if tag else ''
    best_path = f'models/bullet_bill_ppo_v7{suffix}_best.pt'
    last_path = f'models/bullet_bill_ppo_v7{suffix}_last.pt'
    if not HAS_NUMBA:
        print("경고: numba가 없어 계획 시뮬레이션이 매우 느립니다. 'uv pip install numba' 후 실행하세요.")
    os.makedirs('models', exist_ok=True)

    batch = n_envs * n_steps
    n_updates = max(1, total_steps // batch)
    mb_size = batch // minibatches
    print(f"steps={n_updates * batch:,} (업데이트 {n_updates}회 x 배치 {batch}), envs={n_envs}, seed={seed}, "
          f"obs_dim={OBS_DIM}, threads={torch.get_num_threads()}, numba={HAS_NUMBA}")

    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    eval_agent = BulletBillAgent(model)
    eval_seeds = range(10_000, 10_000 + eval_episodes)

    envs = [gym.make(id=ENV_ID, render_mode='none', bgm=False) for _ in range(n_envs)]
    fes = [FeatureExtractor() for _ in range(n_envs)]
    obs_list = []
    s = np.zeros((n_envs, OBS_DIM), np.float32)
    phi = np.zeros(n_envs, np.float32)
    for i, env in enumerate(envs):
        o, inf = env.reset(seed=seed * 1000 + i)
        st, ed, bd = fes[i](o, inf)
        s[i], phi[i] = st, potential(ed, bd)
        obs_list.append(o)

    S = np.zeros((n_steps, n_envs, OBS_DIM), np.float32)
    J = np.zeros((n_steps, n_envs), np.int64)
    M = np.zeros((n_steps, n_envs), np.float32)
    LOGP = np.zeros((n_steps, n_envs), np.float32)
    V = np.zeros((n_steps, n_envs), np.float32)
    R = np.zeros((n_steps, n_envs), np.float32)
    D = np.zeros((n_steps, n_envs), np.float32)

    ep_times, ep_causes = deque(maxlen=200), deque(maxlen=200)
    best, t0, t_log, step = -1.0, time.time(), time.time(), 0
    next_eval = eval_interval
    stats = deque(maxlen=log_every * epochs * minibatches)

    for update in range(1, n_updates + 1):
        for g in opt.param_groups:                        # 학습률 선형 감소
            g['lr'] = lr * (1.0 - (update - 1) / n_updates)

        # ---------- 경험 수집 ----------
        for t in range(n_steps):
            with torch.no_grad():
                st = torch.as_tensor(s)
                j, m, logp = model.act(st)
                v = model.value(st)
            S[t], J[t], M[t], LOGP[t], V[t] = s, j.numpy(), m.numpy(), logp.numpy(), v.numpy()
            for i in range(n_envs):
                env_a = to_env_action(M[t, i], J[t, i])
                on_ground = obs_list[i]['mario'][7] < 0.5
                o, _, term, trunc, inf = envs[i].step(env_a)
                s2, ed2, bd2 = fes[i](o, inf)
                done = term or trunc
                dead = term and inf.get('game_state') == 'game_over'
                phi2 = 0.0 if done else potential(ed2, bd2)
                r = SURVIVE + SHAPE * (GAMMA * phi2 - phi[i])
                if on_ground and env_a[1] > 0:
                    r += JUMP_COST
                if dead:
                    r += DEATH
                R[t, i], D[t, i] = r, float(done)
                if done:
                    ep_times.append(float(inf['time_elapsed']))
                    ep_causes.append(death_cause(o, inf))
                    o, inf = envs[i].reset()
                    fes[i].reset()
                    s2, ed2, bd2 = fes[i](o, inf)
                    phi2 = potential(ed2, bd2)
                obs_list[i], s[i], phi[i] = o, s2, phi2
        step += batch

        # ---------- GAE ----------
        with torch.no_grad():
            last_v = model.value(torch.as_tensor(s)).numpy()
        adv = compute_gae(R, V, D, last_v, GAMMA, lam)
        ret = adv + V

        b_s = torch.as_tensor(S.reshape(-1, OBS_DIM))
        b_j = torch.as_tensor(J.reshape(-1))
        b_m = torch.as_tensor(M.reshape(-1))
        b_logp = torch.as_tensor(LOGP.reshape(-1))
        b_adv = torch.as_tensor(adv.reshape(-1))
        b_ret = torch.as_tensor(ret.reshape(-1))

        # ---------- PPO 업데이트 ----------
        idx = np.arange(batch)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for k in range(0, batch, mb_size):
                mbi = torch.as_tensor(idx[k:k + mb_size])
                logp, ent_j, ent_m, v = model.evaluate(b_s[mbi], b_j[mbi], b_m[mbi])
                ratio = (logp - b_logp[mbi]).exp()
                a = b_adv[mbi]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - clip, 1 + clip)).mean()
                v_loss = 0.5 * ((v - b_ret[mbi]) ** 2).mean()
                loss = pg_loss + vf_coef * v_loss - ent_jump * ent_j.mean() - ent_move * ent_m.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                opt.step()
                stats.append((v_loss.item(), ent_j.mean().item(), ((ratio - 1).abs() > clip).float().mean().item()))

        # ---------- 기록 ----------
        if update % log_every == 0 and ep_times:
            now = time.time()
            speed = log_every * batch / (now - t_log)
            t_log = now
            water = sum(c == 'water' for c in ep_causes) / len(ep_causes) * 100
            vl, ej, cf = np.mean(stats, axis=0)
            p_jump = (J != 0).mean() * 100
            print(f"[{step:>11,}] 최근 {len(ep_times)}판 평균 {np.mean(ep_times):6.1f}s (최대 {np.max(ep_times):6.1f}s) | "
                  f"물 {water:3.0f}% | 점프 선택 {p_jump:4.1f}% | 점프 엔트로피 {ej:.2f} | "
                  f"이동 std {model.move_log_std.exp().item():.2f} | V loss {vl:.3f} | clip {cf:.2f} | "
                  f"{speed:,.0f} steps/s")

        if step >= next_eval or update == n_updates:
            next_eval += eval_interval
            times, causes = evaluate(eval_agent, eval_seeds)
            mark = ''
            if times.mean() > best:
                best = times.mean()
                eval_agent.save(best_path)
                mark = ' ★ best 저장'
            print(f"  [평가 {step:,}] 평균 {times.mean():.1f}s, 중앙값 {np.median(times):.1f}s, "
                  f"최소 {times.min():.1f}s{mark}")
            print(f"    사망 원인: {cause_summary(causes)}")

    eval_agent.save(last_path)
    for env in envs:
        env.close()
    print(f"학습 종료 | best 평가 평균 {best:.1f}s | 소요 {(time.time() - t0) / 60:.1f}분")
    return best_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=10_000_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tag', default='')
    p.add_argument('--envs', type=int, default=16)
    p.add_argument('--eval', type=int, default=0, help='저장된 모델을 N판 평가 (학습 안 함)')
    p.add_argument('--model', default='models/bullet_bill_ppo_v7_best.pt')
    p.add_argument('--check-sim', action='store_true', help='numba/numpy 시뮬레이션 일치 여부와 속도 확인')
    p.add_argument('--threads', type=int, default=0, help='PyTorch CPU 스레드 수 (0이면 기본값)')
    args = p.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.check_sim:
        check_sim()
    elif args.eval:
        agent = BulletBillAgent.load(args.model)
        times, causes = evaluate(agent, range(20_000, 20_000 + args.eval))
        print(f"{args.model} | {args.eval}판")
        print(f"  평균 {times.mean():.1f}s, 중앙값 {np.median(times):.1f}s, "
              f"하위10% {np.percentile(times, 10):.1f}s, 상위10% {np.percentile(times, 90):.1f}s")
        print(f"  2분 이상 {np.mean(times >= 120) * 100:.1f}%")
        print(f"  사망 원인: {cause_summary(causes)}")
    else:
        train(total_steps=args.steps, seed=args.seed, tag=args.tag, n_envs=args.envs)
