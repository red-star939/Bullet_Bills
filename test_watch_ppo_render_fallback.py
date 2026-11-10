"""Headless regression for a non-rendering real-game-shaped environment."""
from __future__ import annotations
import argparse
import sys
import types
from pathlib import Path
import numpy as np
import cv2

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))

class FakeEnv:
    def __init__(self): self.n=0
    def reset(self,seed=None):
        self.n=0
        return self.obs(), {'time_elapsed':0.,'game_state':'playing'}
    def obs(self):
        return {'mario':np.asarray([400.,500.,448.,548.,125.,-30.,0.,0.],np.float32),
                'bullet_bills':np.asarray([[600.,510.,630.,540.,-250.]],np.float32)}
    def step(self, action):
        self.n+=1
        return self.obs(),0.,self.n>=2,False,{'time_elapsed':self.n/15.,'game_state':'game_over' if self.n>=2 else 'playing'}
    def render(self):return None
    def close(self):pass
mod=types.ModuleType('agent_env')
mod.create_env=lambda render_mode='none':FakeEnv()
sys.modules['agent_env']=mod

import watch_ppo
frame=watch_ppo.schematic_frame(FakeEnv().obs(),watch_ppo.load_platforms(BASE/'map_data/object_tiles.csv'))
assert frame.shape==(864,1296,3) and frame.dtype==np.uint8
assert not np.array_equal(frame[501,401],frame[100,100])
views=[]
old_imshow,old_wait,old_close=cv2.imshow,cv2.waitKey,cv2.destroyAllWindows
cv2.imshow=lambda title,image:views.append(image.copy())
cv2.waitKey=lambda ms:-1
cv2.destroyAllWindows=lambda:None
try:
    a=argparse.Namespace(model=str(BASE/'models/mario_ppo_init.pt'),render='rgb_array',
        map=str(BASE/'map_data/object_tiles.csv'),episodes=1,steps=4,seed=3,fps=100000,
        stochastic=False,allow_untrained=True,show_actions=False,print_every=15)
    watch_ppo.run(a)
    assert len(views)==2, len(views)
    views.clear()
    a.render='schematic'
    watch_ppo.run(a)
    assert len(views)==2
finally:
    cv2.imshow,cv2.waitKey,cv2.destroyAllWindows=old_imshow,old_wait,old_close
print('Observation schematic and RGB-None fallback passed!')
