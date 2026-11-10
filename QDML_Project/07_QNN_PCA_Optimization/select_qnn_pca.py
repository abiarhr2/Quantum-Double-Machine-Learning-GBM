"""Nine configurations, evaluated on original data only."""
import argparse
from common import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--epochs',type=int,default=120)
    p.add_argument('--out-dir',default=str(HOME/'results/selection'))
    a=p.parse_args(); out=output(a.out_dir); X,Y,D=load_data(); rows=[]
    for c in (4,6,8,12):
        for l in (3,):
            cfg=dict(components=c,layers=l,epochs=a.epochs,lr=.05)
            for seed in a.seeds:
                path=out/f'c{c}_l{l}_s{seed}.json'
                ident=identity(X,Y,D,cfg)
                if path.exists():
                    r=json.loads(path.read_text())
                    if r['identity']!=ident: raise ValueError('Configuration de reprise incompatible')
                else:
                    r=dict(**cfg,qubits=c,parameters=3*c*l,seed=seed,identity=ident,**evaluate(X,Y,D,cfg,seed))
                    atomic_json(path,r)
                rows.append({k:v for k,v in r.items() if k!='identity'})
    table=pd.DataFrame(rows); table.to_csv(out/'configurations.csv',index=False)
    table['loss']=(table.g_logloss+table.m_logloss)/2
    ranking=table.groupby(['components','layers','epochs','lr']).agg(loss=('loss','mean'),stability=('loss','std'),temps_s=('temps_s','mean')).reset_index()
    ranking['stability']=ranking.stability.fillna(0)
    ranking['score']=ranking.loss+ranking.stability
    ranking=ranking.sort_values(['score','temps_s','components','layers'])
    ranking.to_csv(out/'ranking.csv',index=False)
    best=ranking.iloc[0]
    atomic_json(out/'selected.json',dict(components=int(best.components),layers=int(best.layers),epochs=int(best.epochs),lr=float(best.lr)))
    print(ranking.to_string(index=False))

if __name__=='__main__': main()
