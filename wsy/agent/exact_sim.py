"""
Bullet Bills 정확한 시뮬레이터 (kymnasium 1.2.9의 env.py / objs.py / consts.py 규칙을 프레임 단위로 옮김)

- 마리오: apply_force -> tick_frame (속도 제한, 중력, 정수 이동, 마찰, 블록 충돌 x->y, 지지 판정)
- 킬러: 프레임마다 x += int(v * dt) (블록 무시), 화면 밖으로 나가면 제거
- 충돌 순서: 마리오 이동 -> 킬러 이동 -> 별 충돌 -> 킬러 충돌(밟기 / 무적 / 사망)
- 사망: 킬러 충돌, 또는 마리오 bottom > 864 (화면 아래로 떨어짐)
- 미래에 새로 생성될 킬러는 알 수 없으므로 시뮬레이션하지 않음 (생성 시각만 spawn_frames로 계산)

행동 배열: actions[P, S, 2] = P개 계획 x S스텝 x [이동 힘(-1~1), 점프 힘(0~1)]  (환경의 연속 행동과 동일)
"""
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        return lambda f: f

# ---------- 코드 상수 (consts.py) ----------
FPS, FRAMES_PER_STEP = 60, 4
FRAME_DT = 1.0 / FPS
TILE, MAP_W, MAP_H = 48, 27, 18
SCREEN_W, SCREEN_H = TILE * MAP_W, TILE * MAP_H          # 1296, 864
MARIO_W = MARIO_H = 48
GROUND_MAX_SPEED, AIR_MAX_SPEED, IDLE_THRESHOLD = 750.0, 500.0, 10.0
FRICTION, GRAVITY = 0.90, 2000.0
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


_VY_STEP = 2000.0 * (1.0 / 60.0)                 # 프레임당 중력 증가량 (환경과 같은 64비트 연산)
_VY_STARTS = (0.0, -500.0, -600.0, -750.0, -1000.0)  # 낙하 시작 0, 밟기 반동, 점프(힘 0.6/0.75/1.0, 0.5는 -500)


def restore_vy(vy_obs):
    """관측값(32비트)으로 잘린 낙하 속도를 환경의 64비트 값으로 복원.
    환경의 vy = 시작값 + 33.33...을 프레임마다 더한 값이므로, 시작값과 프레임 수를 역산해 같은 순서로 더함.
    맞는 조합이 없으면 관측값 그대로 사용."""
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


def obs_to_sim(obs, inv_frames=0, vy=None):
    """관측 딕셔너리 -> 시뮬레이터 입력. vy: VyTracker로 복원한 낙하 속도 (없으면 한 시점 복원)"""
    m = obs['mario']
    if vy is None:
        vy = restore_vy(m[5]) if m[7] > 0.5 else float(m[5])
    mario = np.array([m[0], m[1], m[4], vy, 0.0 if m[7] > 0.5 else 1.0], dtype=np.float64)
    bb = np.asarray(obs['bullet_bills'], dtype=np.float64)
    bb = np.ascontiguousarray(bb[np.abs(bb).sum(1) > 0])
    st = np.asarray(obs['stars'], dtype=np.float64)
    st = np.ascontiguousarray(st[np.abs(st).sum(1) > 0])
    return mario, bb, st, int(inv_frames)
