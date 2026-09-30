"""Locked-checkpoint external evaluation. Never trains or adjusts thresholds."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import pandas as pd
from src.utils import ROOT,config
from src.data.kneemri import load_kneemri
from src.inference.predictor import Predictor
from src.evaluation.metrics import calculate
from src.evaluation.plots import curves

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--output');args=parser.parse_args()
    cfg=config(args.config);ext=cfg['external_kneemri'];source=ROOT/ext['root'];out=ROOT/(args.output or ext['results']);out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(ROOT/ext['metadata']); files={p.name:p for p in source.rglob('*.pck')}
    missing=set(meta.volumeFilename)-files.keys()
    if missing:raise ValueError(f'Missing {len(missing)} volumes')
    checkpoint=ROOT/cfg['evaluation']['checkpoint']
    with checkpoint.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    protocol={'checkpoint_sha256':digest,'targets':['acl'],'threshold_source':('Reserved MRNet calibration partition (100 studies); locked before external evaluation' if cfg['evaluation']['threshold_selection']=='reserved_calibration_youden' else 'locked MRNet internal validation'),'primary_mapping':'0 healthy ACL; 1 or 2 ACL injury','unit':'examId; mean score across repeated series; maximum severity','secondary':'complete rupture vs healthy, excluding partial injuries','input':'full sagittal volumes; no ROI crop; missing axial and coronal planes','patient_independence':'not established by examId alone','no_training':True}
    protocol_file=out/'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text())!=protocol:raise ValueError('Protocol changed: use a new results directory')
    protocol_file.write_text(json.dumps(protocol,indent=2))
    predictor=Predictor(cfg=cfg);predpath=out/'volume_predictions.csv'
    rows=pd.read_csv(predpath).to_dict('records') if predpath.exists() else [];done={r['volumeFilename'] for r in rows}
    for _,r in meta.iterrows():
        if r.volumeFilename in done:continue
        volume=load_kneemri(files[r.volumeFilename],cfg['preprocessing'])
        prediction=predictor.predict({'sagittal':volume},str(r.examId))
        score=next(f['probability'] for f in prediction['findings'] if f['key']=='acl')
        rows.append({'examId':int(r.examId),'volumeFilename':r.volumeFilename,'severity':int(r.aclDiagnosis),'probability':score,'pixel_sha256':hashlib.sha256(volume.tobytes()).hexdigest()})
        pd.DataFrame(rows).to_csv(predpath,index=False)
        (out/'status.json').write_text(json.dumps({'state':'running','completed':len(rows),'total':len(meta)}))
        if len(rows)%25==0:print(f'{len(rows)}/{len(meta)}',flush=True)
    volumes=pd.DataFrame(rows); exams=volumes.groupby('examId').agg(severity=('severity','max'),probability=('probability','mean'),series=('volumeFilename','count')).reset_index()
    exams.to_csv(out/'exam_predictions.csv',index=False)
    targets=[{'key':'acl','display':'ACL injury'}];threshold={'acl':predictor.thresholds['acl']}
    y=(exams.severity.values>0).astype(int)[:,None];p=exams.probability.values[:,None]
    metrics=calculate(y,p,targets,threshold);curves(y,p,metrics,targets,out)
    secondary=exams[exams.severity!=1];sm=calculate((secondary.severity.values==2).astype(int)[:,None],secondary.probability.values[:,None],targets,threshold)
    result={'primary':metrics,'complete_vs_healthy':sm,'volumes':len(volumes),'exams':len(exams),'unique_pixel_hashes':int(volumes.pixel_sha256.nunique()),'protocol':protocol}
    (out/'metrics.json').write_text(json.dumps(result,indent=2));(out/'status.json').write_text(json.dumps({'state':'complete','completed':len(rows),'total':len(meta)}))
    print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
