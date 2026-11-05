
import gymnasium as gym
import kymnasium

ENV_ID = "kymnasium/BulletBill-Continuous-Hard-Stage-2"


def create_env(render_mode="none", bgm=False):
    """Bullet Bills 강화학습 환경 생성."""

    env = gym.make(
        id=ENV_ID,
        render_mode=render_mode,
        bgm=bgm
    )

    return env
