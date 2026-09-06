import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
DATA=os.path.expanduser("~/manet-data/dataset.csv.gz")
SEQ=29
BASE=["fwd_seen","fwd_ratio","rreq_recv","no_route","rreq_sent",
      "nb_count","nb_mean_dist","nb_max_dist","speed"]
EW=["fwd_ratio","no_route","nb_max_dist","rreq_sent"]
def load():
    a=pd.read_csv(DATA).sort_values(["run","node_id","time_s"])
    for c in ["rreq_sent","rreq_recv","no_route","fwd_seen"]:
        a[c]=np.log1p(a[c].clip(lower=0))
    g=a.groupby(["run","node_id"])
    for c in EW: a[c+"_ewma"]=g[c].transform(lambda x:x.ewm(alpha=0.3).mean())
    return a, BASE+[c+"_ewma" for c in EW]
def seqs(df,feats,ci):
    X,M,Y=[],[],[]
    for _,g in df.groupby(["run","node_id"],sort=False):
        if g.fwd_seen.sum()==0: continue
        v=g[feats].to_numpy(np.float32); n=min(len(v),SEQ)
        p=np.zeros((SEQ,len(feats)),np.float32); m=np.zeros(SEQ,np.float32)
        p[:n]=v[:n]; m[:n]=1.0
        X.append(p);M.append(m);Y.append(ci[g.cls.iloc[0]])
    return np.stack(X),np.stack(M),np.array(Y)

class CNN1D(nn.Module):
    """Abdan et al. 2021 style: convolutional classifier over node features."""
    def __init__(s,f,n):
        super().__init__()
        s.c=nn.Sequential(nn.Conv1d(f,64,3,padding=1),nn.ReLU(),
                          nn.Conv1d(64,64,3,padding=1),nn.ReLU())
        s.h=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,n))
    def forward(s,x,m):
        o=s.c(x.transpose(1,2)).transpose(1,2)
        mm=m.unsqueeze(-1)
        return s.h((o*mm).sum(1)/mm.sum(1).clamp(min=1.))

class LSTMClf(nn.Module):
    """Venkatasubramanian et al. 2022 style: LSTM over the forwarding series."""
    def __init__(s,f,n):
        super().__init__()
        s.l=nn.LSTM(f,64,num_layers=2,batch_first=True,dropout=0.2)
        s.h=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,n))
    def forward(s,x,m):
        o,_=s.l(x); mm=m.unsqueeze(-1)
        return s.h((o*mm).sum(1)/mm.sum(1).clamp(min=1.))

class GRUClf(nn.Module):
    def __init__(s,f,n):
        super().__init__()
        s.g=nn.GRU(f,64,num_layers=2,batch_first=True,dropout=0.2)
        s.h=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,n))
    def forward(s,x,m):
        o,_=s.g(x); mm=m.unsqueeze(-1)
        return s.h((o*mm).sum(1)/mm.sum(1).clamp(min=1.))

def train_nn(Model,Xtr,Mtr,ytr,nf,nc,ep=30,seed=0):
    torch.manual_seed(seed)
    m=Model(nf,nc); opt=torch.optim.Adam(m.parameters(),1e-3)
    cnt=np.bincount(ytr,minlength=nc).astype(np.float32)
    w=torch.tensor(cnt.sum()/(nc*np.maximum(cnt,1)))
    lf=nn.CrossEntropyLoss(weight=w)
    X,y,M=torch.from_numpy(Xtr),torch.from_numpy(ytr),torch.from_numpy(Mtr)
    for _ in range(ep):
        m.train(); pm=torch.randperm(len(X))
        for i in range(0,len(pm),256):
            j=pm[i:i+256]; opt.zero_grad(); lf(m(X[j],M[j]),y[j]).backward(); opt.step()
    m.eval(); return m

def nn_pred(m,X,M):
    with torch.no_grad():
        lg=m(torch.from_numpy(X),torch.from_numpy(M))
        return lg.argmax(1).numpy(), torch.softmax(lg,1).max(1).values.numpy()

FOLDS=[(range(1,13),range(17,21)),(range(5,17),range(1,5)),
       (range(9,21),range(5,9)),(list(range(1,5))+list(range(13,21)),range(9,13))]

def closed_set():
    a,feats=load(); cl=sorted(a.cls.unique()); ci={c:i for i,c in enumerate(cl)}
    res={}
    for fi,(trs,tes) in enumerate(FOLDS):
        trs,tes=set(trs),set(tes)
        Xtr,Mtr,ytr=seqs(a[a.seed.isin(trs)],feats,ci)
        Xte,Mte,yte=seqs(a[a.seed.isin(tes)],feats,ci)
        sc=StandardScaler().fit(Xtr.reshape(-1,Xtr.shape[-1]))
        f=lambda X:(sc.transform(X.reshape(-1,X.shape[-1])).reshape(X.shape)).astype(np.float32)
        Xtr,Xte=f(Xtr),f(Xte)
        # flat views for the classical learners: mean over valid windows
        def flat(X,M):
            mm=M[:,:,None]; return (X*mm).sum(1)/np.maximum(mm.sum(1),1)
        Ftr,Fte=flat(Xtr,Mtr),flat(Xte,Mte)
        for name,mk in [("Decision tree",lambda:DecisionTreeClassifier(random_state=0,class_weight="balanced")),
                        ("KNN (k=5)",lambda:KNeighborsClassifier(5)),
                        ("SVM (RBF)",lambda:SVC(class_weight="balanced")),
                        ("Random forest",lambda:RandomForestClassifier(300,random_state=0,class_weight="balanced",n_jobs=3))]:
            p=mk().fit(Ftr,ytr).predict(Fte)
            res.setdefault(name,[]).append(f1_score(yte,p,average="macro"))
        for name,Mod in [("CNN-1D",CNN1D),("LSTM",LSTMClf),("GRU (ours)",GRUClf)]:
            m=train_nn(Mod,Xtr,Mtr,ytr,len(feats),len(cl))
            p,_=nn_pred(m,Xte,Mte)
            res.setdefault(name,[]).append(f1_score(yte,p,average="macro"))
        print(f"  fold {fi+1} done")
    print("\n=== Closed-set macro F1, same dataset and split ===")
    for k,v in res.items():
        print(f"  {k:16s} {np.mean(v):.3f} +/- {np.std(v):.3f}")
    return res

def open_set():
    """Every method through the same leave-one-attack-out protocol."""
    a,feats=load()
    print("\n=== Open-set: AUROC separating the held-out attack from normal ===")
    print("  (max-softmax for neural nets, max class probability for the rest)")
    print(f"  {'held out':10s} {'DT':>7s} {'RF':>7s} {'CNN':>7s} {'LSTM':>7s} {'GRU':>7s}")
    summary={}
    for held in ['bha','flood','gha','sink','worm']:
        known=[c for c in ['bha','flood','gha','sink'] if c!=held]+['normal']
        cl=sorted(known); ci={c:i for i,c in enumerate(cl)}
        tr=a[(a.seed<=12)&(a.cls.isin(known))&(a.scenario!=held)]
        te=a[a.seed>16]
        ci_all=dict(ci); ci_all[held]=-1
        for c in a.cls.unique():
            if c not in ci_all: ci_all[c]=-2   # other attacks: excluded below
        Xtr,Mtr,ytr=seqs(tr,feats,ci)
        Xte,Mte,yte=seqs(te,feats,ci_all)
        keep=yte!=-2
        Xte,Mte,yte=Xte[keep],Mte[keep],yte[keep]
        sc=StandardScaler().fit(Xtr.reshape(-1,Xtr.shape[-1]))
        f=lambda X:(sc.transform(X.reshape(-1,X.shape[-1])).reshape(X.shape)).astype(np.float32)
        Xtr,Xte=f(Xtr),f(Xte)
        def flat(X,M):
            mm=M[:,:,None]; return (X*mm).sum(1)/np.maximum(mm.sum(1),1)
        Ftr,Fte=flat(Xtr,Mtr),flat(Xte,Mte)
        isu=(yte==-1).astype(int)
        row={}
        for name,mk in [("DT",lambda:DecisionTreeClassifier(random_state=0,class_weight="balanced")),
                        ("RF",lambda:RandomForestClassifier(300,random_state=0,class_weight="balanced",n_jobs=3))]:
            pr=mk().fit(Ftr,ytr).predict_proba(Fte).max(1)
            row[name]=roc_auc_score(isu,-pr)
        for name,Mod in [("CNN",CNN1D),("LSTM",LSTMClf),("GRU",GRUClf)]:
            m=train_nn(Mod,Xtr,Mtr,ytr,len(feats),len(cl))
            _,conf=nn_pred(m,Xte,Mte)
            row[name]=roc_auc_score(isu,-conf)
        summary[held]=row
        print(f"  {held:10s} "+" ".join(f"{row[k]:7.3f}" for k in ["DT","RF","CNN","LSTM","GRU"]))
    print("\n  Values below 0.500 are worse than chance.")
    n=sum(1 for h in summary for k in summary[h] if summary[h][k]<0.5)
    print(f"  {n} of {len(summary)*5} method-attack pairs fall below chance.")
    return summary

if __name__=="__main__":
    closed_set()
    open_set()
