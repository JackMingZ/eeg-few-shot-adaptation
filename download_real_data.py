"""Verified HTTPS downloads with bounded concurrency and resumable files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests
from scipy.io import whosmat

ROOT=Path(__file__).resolve().parent
BASE='https://lampx.tugraz.at/~bci/database/001-2014/'
DEST=ROOT/'data/MNE-bnci-data/~bci/database/001-2014'


def fetch_segmented(name, workers=4):
    """Finish a mirror file with bounded parallel 1 MiB ranges."""
    DEST.mkdir(parents=True,exist_ok=True)
    path=DEST/name
    if path.exists(): return fetch(name,'nemar')
    manifest=json.loads((ROOT/'references/nemar_manifest.json').read_text(encoding='utf-8'))
    entry=manifest['files'][f'sourcedata/{name}']
    url='https://nemar.s3.us-east-2.amazonaws.com/nm000139/objects/'+entry['key']
    partial=path.with_suffix('.mat.part')
    offset=partial.stat().st_size if partial.exists() else 0
    total=entry['size']
    segments=DEST/(name+'.segments');segments.mkdir(exist_ok=True)
    ranges=[(start,min(start+1024*1024,total)-1) for start in range(offset,total,1024*1024)]

    def chunk(bounds):
        start,end=bounds
        out=segments/f'{start}_{end}'
        if out.exists() and out.stat().st_size==end-start+1: return out
        for attempt in range(4):
            try:
                with requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,30)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}' or len(r.content)!=end-start+1:
                        raise ValueError('Incorrect segment response')
                    out.write_bytes(r.content)
                return out
            except requests.RequestException:
                if attempt==3: raise
        raise RuntimeError('Segment transfer failed')

    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks=list(executor.map(chunk,ranges))
    with partial.open('ab') as f:
        for part in chunks: f.write(part.read_bytes())
    if hashlib.sha256(partial.read_bytes()).hexdigest()!=entry['checksum'].split(':',1)[1]:
        raise ValueError('Assembled file does not match manifest')
    partial.replace(path)
    return fetch(name,'nemar')


def fetch(name, provider='upstream'):
    DEST.mkdir(parents=True,exist_ok=True)
    path=DEST/name
    partial=path.with_suffix('.mat.part')
    manifest_path=ROOT/'references/nemar_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    entry=manifest['files'][f'sourcedata/{name}'] if manifest else None
    url=BASE+name
    if provider=='nemar':
        if entry is None: raise ValueError('NEMAR version manifest is required for mirror downloads')
        url='https://nemar.s3.us-east-2.amazonaws.com/nm000139/objects/'+entry['key']
    if not path.exists() and partial.exists() and entry and partial.stat().st_size==entry['size']:
        if hashlib.sha256(partial.read_bytes()).hexdigest()==entry['checksum'].split(':',1)[1]:
            partial.replace(path)
    if not path.exists():
        for attempt in range(4):
            try:
                offset=partial.stat().st_size if partial.exists() else 0
                total=offset+1
                # Short range requests avoid very slow long-lived transfers.
                while offset<total:
                    headers={'Range':f'bytes={offset}-{offset+1024*1024-1}'}
                    with requests.get(url,headers=headers,stream=True,timeout=(20,30)) as r:
                        r.raise_for_status()
                        ranged=r.status_code==206
                        if ranged and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                            raise ValueError('Unexpected Content-Range')
                        expected=int(r.headers.get('Content-Length',0))
                        total=int(r.headers['Content-Range'].split('/')[-1]) if ranged else expected
                        received=0
                        with partial.open('ab' if ranged else 'wb') as f:
                            for chunk in r.iter_content(8192):
                                f.write(chunk); received+=len(chunk)
                            f.flush()
                        if expected and received!=expected:
                            raise ValueError('Incomplete transfer')
                        offset=partial.stat().st_size
                        if not ranged: break
                    if offset//(10*1024*1024)!=(offset-received)//(10*1024*1024):
                        print(f'PROGRESS {name} {offset}/{total}',flush=True)
                if 'data' not in [x[0] for x in whosmat(partial)]:
                    raise ValueError('Invalid MAT content')
                partial.replace(path)
                break
            except Exception as exc:
                print(f'RETRY {name} {attempt+1}: {exc}',flush=True)
                if attempt==3: raise
                time.sleep(2)
    if 'data' not in [x[0] for x in whosmat(path)]:
        raise ValueError(f'Invalid existing MAT: {path}')
    record={'file':name,'url':url,'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'hash_note':'Locally computed provenance hash, not an upstream signature.'}
    record['transfer_note']='Resumable transfer; prefix may come from upstream or byte-identical NEMAR mirror.'
    if entry:
        expected_hash=entry['checksum'].split(':',1)[1]
        if record['sha256']!=expected_hash or record['bytes']!=entry['size']:
            raise ValueError(f'Checksum/size mismatch against NEMAR manifest: {name}')
        record['verified_against']='NEMAR nm000139 v1.0.2 sourcedata manifest'
    if path.with_suffix('.mat.json').exists():
        previous=json.loads(path.with_suffix('.mat.json').read_text(encoding='utf-8'))
        if previous.get('sha256')==record['sha256']: record['url']=previous['url']
    path.with_suffix('.mat.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'READY {name} bytes={record["bytes"]}',flush=True)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,10)))
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--provider',choices=['upstream','nemar'],default='upstream')
    p.add_argument('--segmented',action='store_true',help='Sequential files, parallel ranges each; NEMAR only')
    p.add_argument('--file-workers',type=int,default=1,help='For segmented mode, distribute the total connection budget across files')
    a=p.parse_args()
    maximum=8 if a.provider=='nemar' else 4
    if not set(a.subjects)<=set(range(1,10)) or not 1<=a.workers<=maximum: p.error('Invalid subjects/workers')
    if not 1<=a.file_workers<=a.workers: p.error('file-workers must be between 1 and workers')
    names=[f'A{s:02d}{session}.mat' for s in a.subjects for session in ['T','E']]
    if a.segmented:
        if a.provider!='nemar': p.error('--segmented requires --provider nemar')
        per_file=max(1,a.workers//a.file_workers)
        with ThreadPoolExecutor(max_workers=a.file_workers) as executor:
            list(executor.map(lambda name:fetch_segmented(name,per_file),names))
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as executor:
            list(executor.map(lambda name:fetch(name,a.provider),names))
