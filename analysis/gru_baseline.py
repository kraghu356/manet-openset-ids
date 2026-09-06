import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

DATA = os.path.expanduser("~/manet-data/dataset.csv.gz")
SEQ_LEN = 29
HIDDEN = 64
EPOCHS = 30
BATCH = 256
LR = 1e-3

torch.manual_seed(0)
np.random.seed(0)

BASE = ["fwd_seen", "fwd_ratio", "rreq_recv", "no_route", "rreq_sent",
        "nb_count", "nb_mean_dist", "nb_max_dist", "speed"]
EWMA = ["fwd_ratio", "no_route", "nb_max_dist", "rreq_sent"]

def load():
    a = pd.read_csv(DATA).sort_values(["run", "node_id", "time_s"])
    g = a.groupby(["run", "node_id"])
    for c in EWMA:
        a[c + "_ewma"] = g[c].transform(lambda x: x.ewm(alpha=0.3).mean())
    return a, BASE + [c + "_ewma" for c in EWMA]

def to_sequences(df, feats, classes):
    X, y, mask = [], [], []
    ci = {c: i for i, c in enumerate(classes)}
    for _, grp in df.groupby(["run", "node_id"], sort=False):
        if grp.fwd_seen.sum() == 0:
            continue
        v = grp[feats].to_numpy(dtype=np.float32)
        n = min(len(v), SEQ_LEN)
        pad = np.zeros((SEQ_LEN, len(feats)), dtype=np.float32)
        m = np.zeros(SEQ_LEN, dtype=np.float32)
        pad[:n] = v[:n]
        m[:n] = 1.0
        X.append(pad)
        mask.append(m)
        y.append(ci[grp.cls.iloc[0]])
    return np.stack(X), np.array(y, dtype=np.int64), np.stack(mask)

class GRUClassifier(nn.Module):
    def __init__(self, n_feat, n_cls, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=2,
                          batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden // 2, n_cls))

    def forward(self, x, mask):
        out, _ = self.gru(x)
        m = mask.unsqueeze(-1)
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled)

def main():
    print("loading ...")
    a, feats = load()
    classes = sorted(a.cls.unique())
    print("classes:", classes)

    Xtr, ytr, mtr = to_sequences(a[a.seed <= 12], feats, classes)
    Xva, yva, mva = to_sequences(a[(a.seed > 12) & (a.seed <= 16)], feats, classes)
    Xte, yte, mte = to_sequences(a[a.seed > 16], feats, classes)
    print(f"sequences  train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}")
    print("train counts:", np.bincount(ytr, minlength=len(classes)))

    sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
    def scale(X):
        s = X.shape
        return sc.transform(X.reshape(-1, s[-1])).reshape(s).astype(np.float32)
    Xtr, Xva, Xte = scale(Xtr), scale(Xva), scale(Xte)

    counts = np.bincount(ytr, minlength=len(classes)).astype(np.float32)
    w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)))
    print("class weights:", w.numpy().round(2))

    model = GRUClassifier(len(feats), len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(weight=w)

    Xtr_t, ytr_t, mtr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(mtr)
    Xva_t, mva_t = torch.from_numpy(Xva), torch.from_numpy(mva)
    Xte_t, mte_t = torch.from_numpy(Xte), torch.from_numpy(mte)

    best_f1, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx], mtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t, mva_t).argmax(1).numpy()
        f1 = f1_score(yva, pv, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  loss {tot/len(perm):.4f}  val macro-F1 {f1:.4f}")

    model.load_state_dict(best_state)
    torch.save(best_state, "/tmp/gru.pt")
    model.eval()
    with torch.no_grad():
        pt = model(Xte_t, mte_t).argmax(1).numpy()
    print()
    print("GRU sequence model, test seeds 17-20")
    print(classification_report(yte, pt, target_names=classes, digits=3))
    print("Random forest, same split: macro 0.918")
    print("  normal 0.997 bha 0.996 gha 0.990 flood 0.983 sink 0.872 worm 0.672")

if __name__ == "__main__":
    main()
