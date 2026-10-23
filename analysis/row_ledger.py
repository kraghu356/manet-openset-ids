"""Run-level ledger from the raw dataset to the node sequences that enter every
evaluation (results/row_ledger.csv), with a summary printed at the end.

For each run: node-window rows recorded, windows in which the node was offered
at least one packet to forward ("active" rows), nodes, nodes active in at least
one window (the evaluated sequences), and the same split by class. A node
that is active in at least one window enters with all of its windows; inactive
windows are kept (their forwarding counts are zero), not removed.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "..", "results", "dataset3.csv.gz"))
d["active_row"] = d.fwd_seen > 0
node = d.groupby(["run", "scenario", "seed", "node_id", "cls"]).agg(rows=("time_s", "size"), active_rows=("active_row", "sum")).reset_index()
node["evaluated"] = node.active_rows > 0
run = node.groupby(["run", "scenario", "seed"]).agg(
    rows=("rows", "sum"), active_rows=("active_rows", "sum"), nodes=("node_id", "size"),
    evaluated_nodes=("evaluated", "sum"),
    attackers=("cls", lambda c: int((c != "normal").sum())),
    evaluated_attackers=("evaluated", lambda e: int(e[node.loc[e.index, "cls"] != "normal"].sum()))).reset_index()
run["windows"] = d.groupby("run").time_s.nunique().reindex(run.run).values
out = os.path.join(HERE, "..", "results", "row_ledger.csv")
run.to_csv(out, index=False)
by_cls = node.groupby("cls").agg(nodes=("node_id", "size"), evaluated=("evaluated", "sum"), rows=("rows", "sum"),
                                rows_of_evaluated=("rows", lambda r: int(r[node.loc[r.index, "evaluated"]].sum())))
print("raw rows", len(d), "active rows", int(d.active_row.sum()), "runs", d.run.nunique())
print("nodes", len(node), "evaluated nodes", int(node.evaluated.sum()),
      "rows of evaluated nodes", int(node[node.evaluated].rows.sum()))
print(by_cls)
print("evaluated attackers per scenario:", run.groupby("scenario").evaluated_attackers.sum().to_dict(),
      "of", run.groupby("scenario").attackers.sum().to_dict())
print("written", out)
