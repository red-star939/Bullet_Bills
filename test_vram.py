import json,os,subprocess,sys,tempfile,unittest,types,importlib.util
from pathlib import Path
import torch
if importlib.util.find_spec('gymnasium') is None: sys.modules['gymnasium']=types.ModuleType('gymnasium')
if importlib.util.find_spec('kymnasium') is None:
    st=types.ModuleType('kymnasium');st.Agent=type('Agent',(),{});sys.modules['kymnasium']=st
import agent_ppo_v10 as v10
from train_models2_v41_ppo import save_pt
HERE=Path(__file__).resolve().parent
class ResidentSmoke(unittest.TestCase):
    def test_cpu_simulation_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); init=root/'init.pt'; out=root/'vram.pt'; log=root/'log'
            save_pt(init,v10.ActorCritic(obs_dim=v10.OBS_DIM,hidden=32),32,{})
            cmd=[sys.executable,str(HERE/'train_models2_v41_ppo_late_vram.py'),'--test-env',
              '--device','cpu','--inference-device','train','--init',str(init),'--out',str(out),
              '--log-dir',str(log),'--envs','3','--batch-steps','4','--epochs','1',
              '--minibatches','2','--eval-episodes','2','--final-episodes','2',
              '--eval-interval','12','--checkpoint-interval','12',
              '--curriculum-min-s','0.07','--curriculum-max-s','0.1','--log-every','1']
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            a=subprocess.run(cmd+['--total-steps','24'],cwd=HERE,env=env,capture_output=True,text=True,timeout=100)
            self.assertEqual(a.returncode,0,a.stdout+a.stderr)
            self.assertIn('rollout_storage=cpu',a.stdout)
            self.assertEqual(json.loads((log/'summary.json').read_text())['completed_steps'],24)
            ck=torch.load(root/'vram_resume.pt',weights_only=False,map_location='cpu')
            self.assertEqual(ck['completed'],24)
            b=subprocess.run(cmd+['--resume',str(root/'vram_resume.pt'),'--total-steps','48'],cwd=HERE,env=env,capture_output=True,text=True,timeout=100)
            self.assertEqual(b.returncode,0,b.stdout+b.stderr)
            ck=torch.load(root/'vram_resume.pt',weights_only=False,map_location='cpu')
            self.assertEqual(ck['completed'],48)
if __name__=='__main__':unittest.main()
