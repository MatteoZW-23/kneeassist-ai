from pathlib import Path
import time,json,subprocess,sys,yaml,hashlib
ROOT=Path(__file__).resolve().parents[1];run=ROOT/'runs/mrnet_fresh_01';result=run/'results';status=run/'completion_status.json'
def update(stage,**kw):status.write_text(json.dumps({'stage':stage,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw},indent=2));print(stage,flush=True)
def command(args):
 subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
def main():
 update('waiting_for_training_and_import')
 while True:
  ready=(result/'final/metrics.json').exists() and (result/'final/example_1130.png').exists()
  # Training announces completion only after all examples are saved.
  train=json.loads((ROOT/'logs/status.json').read_text())
  imported=json.loads((ROOT/'data/external/fastMRI/import_status.json').read_text()).get('state')=='complete'
  if train.get('stage')=='failed':raise RuntimeError('Training failed')
  if ready and train.get('stage')=='completed' and imported:break
  time.sleep(10)
 update('external_kneemri_evaluation')
 command(['-m','src.evaluation.external_kneemri','--config','configs/mrnet_fresh_01.yaml','--output','runs/mrnet_fresh_01/results/external_kneemri'])
 update('external_fastmri_evaluation')
 command(['-m','src.evaluation.external_fastmri','--config','configs/mrnet_fresh_01.yaml','--output','runs/mrnet_fresh_01/results/external_fastmri'])
 update('application_verification')
 cfg=yaml.safe_load((ROOT/'configs/mrnet_fresh_01.yaml').read_text());cfg['lifecycle'].update(state='research_testing_verification',training_enabled=False,inference_enabled=True);cfg['external_kneemri']['results']='runs/mrnet_fresh_01/results/external_kneemri'
 (ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 with (run/'test_results.txt').open('w') as log:subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 update('executing_notebooks')
 command(['execute_notebooks.py'])
 reports={'mrnet':json.loads((result/'final/metrics.json').read_text()),'kneemri':json.loads((result/'external_kneemri/metrics.json').read_text()),'fastmri':json.loads((result/'external_fastmri/metrics.json').read_text())}
 (run/'validation_summary.json').write_text(json.dumps(reports,indent=2))
 cfg['lifecycle']['state']='ready_for_research_testing';(ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 update('ready_for_research_testing',checkpoint=cfg['evaluation']['checkpoint'],clinical_ready=False)
if __name__=='__main__':
 try:main()
 except Exception as error:
  update('failed',error=str(error));cfg=yaml.safe_load((ROOT/'config.yaml').read_text());cfg['lifecycle']['inference_enabled']=False;(ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));raise
