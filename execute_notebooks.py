from pathlib import Path
import json,time,nbformat
from nbclient import NotebookClient
root=Path.cwd();results=[]
for path in sorted((root/'notebooks').glob('*.ipynb')):
 start=time.time();print('Executing',path.name,flush=True)
 notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
 client=NotebookClient(notebook,timeout=1200,kernel_name='kneeassist-ai',resources={'metadata':{'path':str(root)}})
 client.execute()
 nbformat.write(notebook,path)
 results.append({'notebook':path.name,'executed_code_cells':sum(c.cell_type=='code' for c in notebook.cells),'seconds':round(time.time()-start,1),'passed':True})
 (root/'results/notebook_execution.json').write_text(json.dumps(results,indent=2))
 print('Passed',path.name,flush=True)
