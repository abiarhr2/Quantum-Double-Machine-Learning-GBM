"""One atomic file and exclusive lock per iteration, independent of batch boundaries."""
import argparse
import subprocess
import sys
from common import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--start',type=int,default=0); p.add_argument('--end',type=int,default=1000)
    p.add_argument('--skip-original',action='store_true')
    p.add_argument('--config',default=str(HOME/'results/selection/selected.json'))
    p.add_argument('--out-dir',default=str(HOME/'results/bootstrap'))
    a=p.parse_args()
    if not 0<=a.start<a.end<=1000: p.error('0 <= start < end <= 1000')
    out=output(a.out_dir); cfg=json.loads(Path(a.config).read_text()); X,Y,D=load_data()
    ident=identity(X,Y,D,cfg)
    # Freeze configuration atomically without replacing an existing run manifest.
    try:
        with (out/'identity.json').open('x') as f: json.dump(ident,f,indent=2)
    except FileExistsError:
        if json.loads((out/'identity.json').read_text())!=ident: raise ValueError('Reprise incompatible')
    if not a.skip_original and not (out/'original.json').exists():
        atomic_json(out/'original.json',evaluate(X,Y,D,cfg))
    for b in range(a.start,a.end):
        path=out/f'iteration_{b:04d}.json'
        if path.exists(): continue
        lock=path.with_suffix('.lock')
        try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError: raise RuntimeError(f'Iteration verrouillee: {lock}; verifier job actif avant suppression')
        os.close(fd)
        try:
            idx=indices(b,len(Y)); t=time.perf_counter()
            r=dict(b=b,idx_hash=digest(idx),valid=valid(idx,D),identity=ident)
            if r['valid']:
                r.update(evaluate(X[idx],Y[idx],D[idx],cfg,seed=b,groups=idx))
            r['temps_s']=time.perf_counter()-t
            atomic_json(path,r)
        finally: lock.unlink(missing_ok=True)
    # All array workers can trigger this; comparison itself uses an exclusive lock.
    if all((out/f'iteration_{b:04d}.json').exists() for b in range(1000)):
        subprocess.run([sys.executable,str(HOME/'compare_qnn_pca.py'),'--pca-dir',str(out)],check=True)

if __name__=='__main__': main()
