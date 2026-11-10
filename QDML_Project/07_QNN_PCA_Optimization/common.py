"""PCA QNN, with preprocessing fitted exclusively inside each training fold."""
import ast
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import log_loss, brier_score_loss

HOME=Path(__file__).resolve().parent
ROOT=HOME.parent
OLD=ROOT/'Projet_qdml_bootstrap/04_MLP_Optimized_vs_QNN_NonOptimized_Bootstrap/resultats_bootstrap_MLP_Optimized_Not_QNN_Optimized'
OPT=ROOT/'Projet_qdml_bootstrap/05_QNN_Optimized_Paired_Bootstrap/resultats_bootstrap_QNN_OPT'
SOURCE=OPT.parent/'05_QNN_Optimized_Paired_Bootstrap.py'

def output(path):
    p=Path(path).resolve()
    if not p.is_relative_to(HOME): raise ValueError('Sorties autorisees uniquement dans experience 07')
    p.mkdir(parents=True,exist_ok=True)
    return p

def atomic_json(path,value):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def load_data(path=None):
    tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SELECTED_RADIO' for t in n.targets))
    df=pd.read_excel(path or ROOT/'data/GBM_GBS.xlsx',header=3)
    df.columns=df.columns.str.strip()
    df=df.dropna(subset=['name']).dropna(axis=1,how='all').reset_index(drop=True)
    return df[names].apply(pd.to_numeric,errors='coerce').values,df['name'].astype(str).str.endswith('GSM').astype(int).values,(df.Edema=='Y').astype(int).values

def indices(b,n): return np.random.RandomState(b).choice(n,n,replace=True)
def digest(idx): return hashlib.md5(np.asarray(idx,dtype=np.int64).tobytes()).hexdigest()[:12]
def valid(idx,D): return all(len(np.unique(idx[D[idx]==c]))>=5 for c in (0,1))
def splits(X,D,seed,groups=None):
    cls=StratifiedKFold if groups is None else StratifiedGroupKFold
    return list(cls(n_splits=5,shuffle=True,random_state=seed).split(X,D,groups))
def fold_hash(folds,n):
    labels=np.empty(n,dtype=np.int64)
    for k,(_,te) in enumerate(folds): labels[te]=k
    return digest(labels)

def fit_predict(Xtr,ytr,Xte,cfg,seed):
    import pennylane as qml
    from pennylane import numpy as pnp
    prep=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),PCA(n_components=cfg['components'],svd_solver='full'),MinMaxScaler(feature_range=(-np.pi,np.pi),clip=True))
    train=prep.fit_transform(Xtr); test=prep.transform(Xte)
    q=cfg['components']; layers=cfg['layers']
    dev=qml.device('lightning.qubit',wires=q)
    @qml.qnode(dev,diff_method='adjoint')
    def circuit(w,x):
        qml.AngleEmbedding(x,wires=range(q),rotation='Y')
        qml.StronglyEntanglingLayers(w,wires=range(q))
        return qml.expval(qml.PauliZ(0))
    def predict(w,x): return (1-pnp.array([circuit(w,row) for row in x]))/2
    w=pnp.array(.1*np.random.RandomState(seed).randn(layers,q,3),requires_grad=True)
    target=pnp.array(ytr,dtype=float)
    def cost(w):
        p=pnp.clip(predict(w,train),1e-6,1-1e-6)
        return -pnp.mean(target*pnp.log(p)+(1-target)*pnp.log(1-p))
    opt=qml.AdamOptimizer(stepsize=cfg['lr']); t=time.perf_counter()
    for _ in range(cfg['epochs']): w=opt.step(cost,w)
    return np.asarray(predict(w,test),dtype=float),time.perf_counter()-t,float(prep.named_steps['pca'].explained_variance_ratio_.sum())

def evaluate(X,Y,D,cfg,seed=42,groups=None):
    start=time.perf_counter(); folds=splits(X,D,seed,groups)
    g=np.zeros(len(Y)); m=g.copy(); durations=[]; variances=[]
    for tr,te in folds:
        if groups is not None and np.intersect1d(groups[tr],groups[te]).size: raise ValueError('Fuite patients')
        for target,pred in ((Y,g),(D,m)):
            pred[te],dt,var=fit_predict(X[tr],target[tr],X[te],cfg,seed)
            durations.append(dt); variances.append(var)
    if not np.isfinite([g,m]).all(): raise ValueError('Predictions non finies')
    dr=D-np.clip(m,.01,.99); yr=Y-g; denom=np.sum(dr**2)
    if denom<=1e-12: raise ValueError('Denominateur nul')
    theta=np.sum(dr*yr)/denom
    se=np.sqrt(np.mean(((yr-theta*dr)*dr)**2)/np.mean(dr**2)**2/len(Y))
    return dict(theta=float(theta),se=float(se),ic_low=float(theta-1.96*se),ic_high=float(theta+1.96*se),g_logloss=log_loss(Y,g,labels=[0,1]),m_logloss=log_loss(D,m,labels=[0,1]),g_brier=brier_score_loss(Y,g),m_brier=brier_score_loss(D,m),training_s=sum(durations),temps_s=time.perf_counter()-start,variance_explained=float(np.mean(variances)),fold_hash=fold_hash(folds,len(Y)))

def identity(X,Y,D,cfg):
    import sklearn,pennylane
    return dict(config=cfg,data_hash=hashlib.sha256(pd.DataFrame(X).to_csv(index=False).encode()+Y.tobytes()+D.tobytes()).hexdigest(),sklearn=sklearn.__version__,pennylane=pennylane.__version__,n=len(Y),boot_seed=0,folds=5)
