from pathlib import Path
import json,urllib.request,hashlib,time,datetime
ROOT=Path(__file__).resolve().parent
folder=ROOT/'data/external/KneeMRI'
record=json.loads((folder/'record.json').read_text())
status_path=folder/'transfer_status.json'
def status(**kwargs):
 status_path.write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs},indent=2))
for item in sorted(record['files'],key=lambda x:x['size']):
 dest=folder/item['key']; partial=dest.with_suffix(dest.suffix+'.part'); expected=item['checksum'].split(':')[1]
 if dest.exists():
  with dest.open('rb') as f: valid=hashlib.file_digest(f,'md5').hexdigest()==expected
  if valid:continue
 for attempt in range(1,6):
  try:
   offset=partial.stat().st_size if partial.exists() else 0
   request=urllib.request.Request(item['links']['self'],headers={'Range':f'bytes={offset}-'} if offset else {})
   with urllib.request.urlopen(request,timeout=120) as response:
    if offset and response.status!=206:offset=0
    if response.status==206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Invalid resume response')
    received=offset; last=0
    with partial.open('ab' if offset else 'wb') as out:
     while chunk:=response.read(1024*1024):
      out.write(chunk);received+=len(chunk)
      if time.monotonic()-last>5:
       out.flush();status(state='downloading',file=item['key'],received_bytes=received,total_bytes=item['size'],percent=round(received/item['size']*100,2),attempt=attempt);last=time.monotonic()
   if received!=item['size']:raise ValueError('Incomplete file')
   status(state='verifying_checksum',file=item['key'])
   with partial.open('rb') as f:actual=hashlib.file_digest(f,'md5').hexdigest()
   if actual!=expected:raise ValueError('Checksum mismatch; file remains .part and must not be used')
   partial.replace(dest);print('Verified',item['key'],flush=True);break
  except Exception as e:
   status(state='retrying' if attempt<5 else 'failed',file=item['key'],error=str(e),attempt=attempt)
   if attempt==5:raise
   time.sleep(15)
status(state='complete',all_source_checksums_verified=True,files=[f['key'] for f in record['files']])
