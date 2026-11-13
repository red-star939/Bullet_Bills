import importlib.util, os, subprocess, sys, tempfile, types, unittest
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
if importlib.util.find_spec('gymnasium') is None: sys.modules['gymnasium']=types.ModuleType('gymnasium')
if importlib.util.find_spec('kymnasium') is None:
    stub=types.ModuleType('kymnasium'); stub.Agent=type('Agent', (), {}); sys.modules['kymnasium']=stub
import agent_ppo_v10 as v10
from train_models2_v41_ppo import save_pt
class TestLate(unittest.TestCase):
  def test_warmup_and_resume(self):
    with tempfile.TemporaryDirectory() as td:
      root=Path(td); init=root/'init.pt'; out=root/'late.pt'; log=root/'log'
      save_pt(init,v10.ActorCritic(obs_dim=v10.OBS_DIM,hidden=32),32,{})
      cmd=[sys.executable,str(HERE/'train_models2_v41_ppo_late.py'),'--test-env','--init',str(init),'--out',str(out),'--log-dir',str(log),'--envs','3','--batch-steps','4','--epochs','1','--minibatches','2','--eval-episodes','2','--final-episodes','2','--eval-interval','12','--checkpoint-interval','12','--curriculum-min-s','0.07','--curriculum-max-s','0.1','--log-every','1']
      env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
      a=subprocess.run(cmd+['--total-steps','24'],cwd=HERE,env=env,capture_output=True,text=True,timeout=100)
      self.assertEqual(a.returncode,0,a.stdout+a.stderr)
      self.assertTrue((log/'summary.json').is_file())
      import json
      summary=json.loads((log/'summary.json').read_text())
      self.assertGreater(summary['late_curriculum']['warmup']['env_steps'],0)
      self.assertGreater(summary['late_curriculum']['warmup']['completed'],0)
      ck=torch.load(root/'late_resume.pt',weights_only=False,map_location='cpu')
      self.assertEqual(ck['completed'],24)
      b=subprocess.run(cmd+['--resume',str(root/'late_resume.pt'),'--total-steps','48'],cwd=HERE,env=env,capture_output=True,text=True,timeout=100)
      self.assertEqual(b.returncode,0,b.stdout+b.stderr)
      ck2=torch.load(root/'late_resume.pt',weights_only=False,map_location='cpu')
      self.assertEqual(ck2['completed'],48)
  def test_unreachable_target_fails_cleanly(self):
    with tempfile.TemporaryDirectory() as td:
      root=Path(td); init=root/'init.pt'; save_pt(init,v10.ActorCritic(obs_dim=v10.OBS_DIM,hidden=32),32,{})
      cmd=[sys.executable,str(HERE/'train_models2_v41_ppo_late.py'),'--test-env','--init',str(init),'--out',str(root/'best.pt'),'--log-dir',str(root/'log'),'--envs','2','--batch-steps','2','--epochs','1','--minibatches','2','--eval-episodes','1','--final-episodes','1','--curriculum-min-s','1','--curriculum-max-s','1','--warmup-max-attempts','2','--total-steps','4']
      env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
      result=subprocess.run(cmd,cwd=HERE,env=env,capture_output=True,text=True,timeout=100)
      self.assertNotEqual(result.returncode,0)
      self.assertIn('failed to survive to target',result.stderr)
if __name__=='__main__': unittest.main()
