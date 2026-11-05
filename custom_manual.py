
import numpy as np
import pygame

from kymnasium.envs.bullet_bill import ManualPlayWrapper


class CustomManualPlayWrapper(ManualPlayWrapper):
    # 즉시 반응하도록 최소 점프 강도 설정
    JUMP_FORCE = 0.5

    def _current_action(self):
        keys = pygame.key.get_pressed()

        horizontal = (
            int(keys[pygame.K_RIGHT])
            - int(keys[pygame.K_LEFT])
        )

        jump = self.JUMP_FORCE if keys[pygame.K_SPACE] else 0.0

        return np.array(
            [float(horizontal), float(jump)],
            dtype=np.float32
        )
