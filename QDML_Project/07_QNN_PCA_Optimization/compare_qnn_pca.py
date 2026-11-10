"""Read-only historical comparison. Never claim unrecorded pairing evidence."""
import argparse
from itertools import combinations
from common import *

def load_boot(path,cols):
    files=sorted(path.glob('boot_*_*.csv'))
    if not files: raise ValueError(f'Aucun bootstrap: {path}')
    df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    if df.b.isna().any() or (df.b%1!=0).any(): raise ValueError('Indices b invalides')
    if df.groupby('b')[cols].nunique(dropna=False).gt(1).any().any(): raise ValueError('Doublons contradictoires')
    return df.drop_duplicates('b').set_index('b').sort_index()

def summary(s):
    s=s.replace([np.inf,-np.inf],np.nan).dropna()
    return dict(n_valid=len(s),mean=s.mean(),std=s.std(),ci_low=s.quantile(.025),ci_high=s.quantile(.975))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pca-dir',default=str(HOME/'results/bootstrap'))
    p.add_argument('--out-dir',default=str(HOME/'results/comparison'))
    p.add_argument('--expected',type=int,default=1000)
    a=p.parse_args(); out=output(a.out_dir); newdir=Path(a.pca_dir)
    lock=out/'comparison.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: print('Comparaison deja active'); return
    os.close(fd)
    try: compare(a,out,newdir)
    finally: lock.unlink(missing_ok=True)

def compare(a,out,newdir):
    old=load_boot(OLD,['NN','QNN']); opt=load_boot(OPT,['QNN_OPT','idx_hash'])
    records=[json.loads(f.read_text()) for f in sorted(newdir.glob('iteration_*.json'))]
    if not records: raise ValueError('Aucun resultat PCA')
    new=pd.DataFrame(records).set_index('b')
    if new.index.has_duplicates: raise ValueError('Doublons PCA')
    X,Y,D=load_data(); cfg=json.loads((newdir/'identity.json').read_text())
    audit=[]; verified=[]
    optcfg=json.loads((OPT/'config_QNN_OPT.json').read_text())
    for b,r in new.iterrows():
        b=int(b); idx=indices(b,len(Y)); fh=fold_hash(splits(X[idx],D[idx],b,idx),len(Y)) if valid(idx,D) else None
        ok=(r.idx_hash==digest(idx) and r['identity']==cfg and bool(r['valid'])==valid(idx,D) and (not valid(idx,D) or r.get('fold_hash')==fh))
        optok=(b in opt.index and str(opt.loc[b,'idx_hash'])==digest(idx) and optcfg.get('n')==len(Y) and optcfg.get('boot_seed')==0 and optcfg.get('fold_hash')=='a56f4caa2b1e')
        audit.append(dict(b=b,pca_verified=ok,opt_sample_verified=optok,legacy04_verified=False,reason='04: aucun manifeste des indices/plis; 05: empreinte globale sur b=0,1,2 seulement'))
        # Per-iteration folds are not recorded by experiment 05 either.
        verified.append(False)
    pd.DataFrame(audit).to_csv(out/'pairing_audit.csv',index=False)
    boot=old[['NN','QNN']].join(opt[['QNN_OPT']],how='outer').join(new[['theta']].rename(columns={'theta':'QNN_PCA'}),how='outer') if 'theta' in new else old[['NN','QNN']].join(opt[['QNN_OPT']],how='outer').assign(QNN_PCA=np.nan)
    models=list(boot); rows=[]
    originals={}
    for path in (OLD/'resultats_originaux.csv',OPT/'resultats_originaux_QNN_OPT.csv'):
        if path.exists():
            frame=pd.read_csv(path,index_col=0)
            originals.update(frame.to_dict('index'))
    if (newdir/'original.json').exists(): originals['QNN_PCA']=json.loads((newdir/'original.json').read_text())
    for m in models:
        frame=old if m in ('NN','QNN') else opt if m=='QNN_OPT' else new
        orig=originals.get(m,{})
        times=frame.get('temps_s',pd.Series(dtype=float))
        rows.append(dict(model=m,**summary(boot[m]),theta_original=orig.get('theta'),original_ci_low=orig.get('IC_low',orig.get('ic_low')),original_ci_high=orig.get('IC_high',orig.get('ic_high')),original_s=orig.get('temps_s'),mean_iteration_s=times.mean(),total_s=times.sum(min_count=1),recorded=len(frame),missing=len(set(range(a.expected))-set(frame.index)),**{k:orig.get(k) for k in ('g_logloss','m_logloss','g_brier','m_brier')}))
    table=pd.DataFrame(rows).set_index('model')
    baseline=table.loc['QNN_OPT','mean_iteration_s']
    table['time_reduction_vs_opt_pct']=100*(1-table.mean_iteration_s/baseline) if baseline>0 else np.nan
    table.to_csv(out/'comparison_models.csv'); boot.to_csv(out/'bootstrap_thetas.csv')
    # Keep all six requested contrasts explicitly visible, with reasons rather than invented intervals.
    pd.DataFrame([dict(contrast=f'{x} - {y}',n_verified=0,mean=np.nan,ci_low=np.nan,ci_high=np.nan,status='BLOCKED: historical per-iteration pairing provenance unavailable') for x,y in combinations(models,2)]).to_csv(out/'paired_differences.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for m in models:
        s=boot[m].dropna()
        if len(s): axes[0].hist(s,bins=30,alpha=.4,label=m)
    axes[0].legend(); axes[0].set_title('Theta bootstrap (non apparie)')
    for i,m in enumerate(models):
        r=table.loc[m]
        axes[1].plot([r.ci_low,r.ci_high],[i,i]); axes[1].plot(r['mean'],i,'o')
    axes[1].set_yticks(range(4),models); axes[1].set_title('IC percentile 95 %')
    table.mean_iteration_s.plot.bar(ax=axes[2]); axes[2].set_ylabel('secondes / iteration (NA = absent)')
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)
    (out/'limitations.txt').write_text('Appariement historique non certifiable: aucun indice/plis par iteration dans 04; seulement trois plis de reference dans 05. Differences appariees bloquees. Temps bootstrap NN/QNN et metriques auxiliaires historiques absents. Un theta plus eleve ne prouve ni performance superieure ni causalite. Voir pairing_audit.csv.\n',encoding='utf-8')
    print(table.to_string()); print('Differences appariees bloquees: provenance historique insuffisante.')

if __name__=='__main__': main()
