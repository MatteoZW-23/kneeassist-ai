"""Evaluate reviewed fastMRI files against limited meniscus annotation reference."""
from pathlib import Path
import argparse,json,hashlib
import pandas as pd,numpy as np,h5py
from src.utils import ROOT,config
from src.inference.predictor import Predictor
from src.evaluation.metrics import calculate
from src.evaluation.plots import curves

def main():
 p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--output',required=True);args=p.parse_args();cfg=config(args.config)
 folder=ROOT/'data/external/fastMRI';status=json.loads((folder/'import_status.json').read_text())
 if status['state']!='complete' or not status.get('xz_integrity_verified'):raise ValueError('Import integrity verification is incomplete')
 manifest=pd.read_csv(folder/'image_manifest.csv');labels=pd.read_csv(ROOT/cfg['external_fastmri']['annotations']);reviewed=set((ROOT/cfg['external_fastmri']['reviewed_files']).read_text().splitlines())
 out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
 with (ROOT/cfg['evaluation']['checkpoint']).open('rb') as f:ck=hashlib.file_digest(f,'sha256').hexdigest()
 protocol={'checkpoint_sha256':ck,'target':'meniscus','positive_reference':'At least one exact Meniscus Tear annotation','negative_reference':'Reviewed file without Meniscus Tear annotation; limited non-exhaustive reference','input':'coronal reconstruction_rss only; no annotation-informed crops','unit':'file; patient linkage unavailable','training_on_this_dataset':False,'thresholds':('Reserved MRNet calibration partition (100 studies); locked before external evaluation' if cfg['evaluation']['threshold_selection']=='reserved_calibration_youden' else 'MRNet internal validation only'),'scope':'exploratory external evaluation, not clinical ground truth'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));predictor=Predictor(cfg=cfg);rows=[]
 for r in manifest.to_dict('records'):
  if r['file_id'] not in reviewed:continue
  if r['acquisition'] not in ['CORPD_FBK','CORPDFS_FBK']:raise ValueError('Unverified MRI orientation')
  with h5py.File(folder/'images'/r['filename'],'r') as f:volume=f['reconstruction_rss'][()]
  # Full native image orientation, consistent with training preprocessing; boxes not used.
  pred=predictor.predict({'coronal':volume},r['file_id']);score=next(x['probability'] for x in pred['findings'] if x['key']=='meniscus')
  positive=bool(((labels.file==r['file_id'])&(labels.label=='Meniscus Tear')).any())
  rows.append({'file_id':r['file_id'],'reference':int(positive),'probability':score})
  if len(rows)%25==0:print('fastMRI evaluated',len(rows),flush=True)
 frame=pd.DataFrame(rows);frame.to_csv(out/'predictions.csv',index=False);y=frame.reference.to_numpy()[:,None];scores=frame.probability.to_numpy()[:,None]
 targets=[{'key':'meniscus','display':'Meniscus annotation reference'}];metrics=calculate(y,scores,targets,{'meniscus':predictor.thresholds['meniscus']});curves(y,scores,metrics,targets,out)
 report={'files':len(frame),'excluded_unreviewed':len(manifest)-len(frame),'metrics':metrics,'protocol':protocol};(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
