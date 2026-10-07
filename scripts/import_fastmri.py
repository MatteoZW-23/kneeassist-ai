from pathlib import Path
import argparse
import tarfile,lzma,shutil,json,hashlib
import h5py,numpy as np,pandas as pd
parser = argparse.ArgumentParser(description='Import a fastMRI single-coil archive into the external evaluation folder.')
parser.add_argument('--archive', type=Path, required=True, help='Path to the original knee_singlecoil_val.tar.xz source archive.')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
arc = args.archive.expanduser().resolve()
if not arc.is_file():
    raise FileNotFoundError(f'fastMRI source archive was not found: {arc}')
out = root / 'data/external/fastMRI'
out.mkdir(exist_ok=True)
vol = out / 'images'
vol.mkdir(exist_ok=True)
labels=pd.read_csv(root/'data/external/fastMRI-plus/knee.csv');reviewed=set((root/'data/external/fastMRI-plus/knee_file_list.csv').read_text().splitlines());rows=[]
with lzma.open(arc,'rb') as stream:
 with tarfile.open(fileobj=stream,mode='r|') as tar:
  for member in tar:
   if member.isdir():continue
   if not member.isfile() or not member.name.endswith('.h5'):raise ValueError('Unexpected archive member')
   name=Path(member.name).name
   if name in {r['filename'] for r in rows}:raise ValueError('Duplicate filename')
   dest=vol/name
   if not dest.exists() or dest.stat().st_size!=member.size:
    with tar.extractfile(member) as src,dest.with_suffix('.part').open('wb') as target:shutil.copyfileobj(src,target,8*1024*1024)
    dest.with_suffix('.part').replace(dest)
   with h5py.File(dest,'r') as f:
    key='reconstruction_rss' if 'reconstruction_rss' in f else 'reconstruction_esc';image=f[key][()]
    if image.ndim!=3 or not np.isfinite(image).all():raise ValueError('Invalid image')
    identifier=dest.stem;matched=labels[labels.file==identifier]
    rows.append({'filename':name,'file_id':identifier,'reviewed':identifier in reviewed,'annotation_rows':len(matched),'labels':'|'.join(sorted(set(matched.label))),'acquisition':str(f.attrs.get('acquisition','unknown')),'reconstruction':key,'shape':str(image.shape),'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest()})
   pd.DataFrame(rows).to_csv(out/'image_manifest.csv',index=False)
   (out/'import_status.json').write_text(json.dumps({'state':'importing','volumes':len(rows)}))
   if len(rows)%10==0:print('Imported',len(rows),flush=True)
 while stream.read(8*1024*1024):pass
summary={'state':'complete','volumes':len(rows),'reviewed_volumes':sum(r['reviewed'] for r in rows),'volumes_with_annotations':sum(r['annotation_rows']>0 for r in rows),'xz_integrity_verified':True,'training_started':False,'label_mapping':'pending review; ACL sprain is not automatically MRNet tear'}
(out/'import_status.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
