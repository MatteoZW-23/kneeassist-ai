"""Double-click launcher: starts the local app and opens its browser page."""
import json
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parent

def main():
    cfg=yaml.safe_load((ROOT/'config.yaml').read_text())
    host=cfg['ui']['host'];port=cfg['ui']['port'];url=f'http://{host}:{port}'
    try:
        with urllib.request.urlopen(url+'/_stcore/health',timeout=2) as response:
            if response.status==200:webbrowser.open(url);return
    except Exception:pass
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/app.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-m','streamlit','run',str(ROOT/'app.py'),
            '--server.address',host,'--server.port',str(port),'--server.headless','true'],
            cwd=ROOT,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    (ROOT/'logs/app_process.json').write_text(json.dumps({'pid':process.pid,'url':url}))
    for _ in range(60):
        if process.poll() is not None:raise RuntimeError('Application stopped. See logs/app.log for details.')
        try:
            with urllib.request.urlopen(url+'/_stcore/health',timeout=1) as response:
                if response.status==200:webbrowser.open(url);return
        except Exception:time.sleep(1)
    raise RuntimeError('Application startup timed out. See logs/app.log.')

if __name__=='__main__':main()
