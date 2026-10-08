#!/usr/bin/env python3
"""Small, record-held-out BIDMC PPG temporal-organization feasibility pilot."""
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

def split_group(rid, assignment=None):
    if assignment is not None: return assignment[rid]
    h = int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) % 10
    return "train" if h < 6 else ("val" if h < 8 else "test")

def load_record(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(subset=["Time [s]", "PLETH"])
    t, x = df["Time [s]"].to_numpy(float), df["PLETH"].to_numpy(float)
    if len(t) < 50000 or np.nanmax(np.diff(t)) > .02:
        return None
    fs = 1 / np.median(np.diff(t))
    b, a = butter(3, [0.5/(fs/2), 8/(fs/2)], btype="band")
    z = filtfilt(b, a, x)
    peaks, _ = find_peaks(z, distance=int(.30*fs), prominence=max(np.std(z)*.15, 1e-4))
    rr = np.diff(t[peaks])
    valid = (rr >= .30) & (rr <= 2.0)
    rr = rr[valid]
    if len(rr) < 20: return None
    # Future ordered target: two successive future RR changes agree in sign.
    rows=[]
    for i in range(8, len(rr)-3):
        past = rr[i-8:i]
        d1, d2 = rr[i+1]-rr[i], rr[i+2]-rr[i+1]
        if min(abs(d1), abs(d2)) < .015 or d1*d2 == 0: continue
        y = int(d1*d2 > 0)
        rate = np.mean(past[-4:]); slope = past[-1]-past[-2]
        quality = float(np.median(np.abs(past-np.median(past))))
        state = np.array([rate, past[-1], slope, quality])
        rows.append((state, np.sort(past), past.copy(), y))
    return dict(record=path.stem.split("_")[1], fs=float(fs), samples=len(t), peaks=len(peaks),
                valid_rr_frac=float(np.mean(valid)), rr_count=len(rr), rows=rows,
                rr_median=float(np.median(rr)), rr_mad=float(np.median(np.abs(rr-np.median(rr)))))

def fit_eval(Xtr, ytr, Xte, yte):
    sc=StandardScaler().fit(Xtr); clf=LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Xtr),ytr)
    p=clf.predict_proba(sc.transform(Xte))[:,1]
    return dict(brier=float(brier_score_loss(yte,p)), log_loss=float(log_loss(yte,p,labels=[0,1])),
                auc=float(roc_auc_score(yte,p)) if len(np.unique(yte))==2 else None,
                n=int(len(yte)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    audits=[]; allrows=[]
    for p in sorted(Path(args.data).glob("bidmc_*_Signals.csv")):
        try: r=load_record(p)
        except Exception as e: print("skip",p,e); continue
        if r is None: continue
        audits.append({k:v for k,v in r.items() if k!="rows"})
        if r["valid_rr_frac"] < 0.90:
            continue
        for state, unordered, ordered, y in r["rows"]: allrows.append((r["record"],state,unordered,ordered,y))
    pd.DataFrame(audits).to_csv(out/"record_audit.csv",index=False)
    if not allrows: raise SystemExit("No usable recordings")
    rec=np.array([x[0] for x in allrows]); states=np.stack([x[1] for x in allrows]); unord=np.stack([x[2] for x in allrows]); ordered=np.stack([x[3] for x in allrows]); y=np.array([x[4] for x in allrows])
    ids=sorted({x[0] for x in allrows}, key=int); n=len(ids); assignment={rid:("train" if j < int(.6*n) else ("val" if j < int(.8*n) else "test")) for j,rid in enumerate(ids)}
    groups=np.array([split_group(x,assignment) for x in rec]);
    counts={g:int(np.sum(groups==g)) for g in ["train","val","test"]}
    if counts["test"]==0 or len(np.unique(y[groups=="test"]))<2: raise SystemExit("Degenerate held-out target")
    # A: state; B: state plus sorted local history; C: state plus ordered local history.
    feats={"A_state":states,"B_unordered":np.c_[states,unord],"C_ordered":np.c_[states,ordered]}
    results={"records":len(ids),"audited_records":len(audits),"excluded_low_detector_validity":[a["record"] for a in audits if a["record"] not in ids],"record_ids":ids,"target": "y=1 iff (RR[i+1]-RR[i]) and (RR[i+2]-RR[i+1]) have the same nonzero sign; both absolute changes >=0.015 s; otherwise excluded", "target_positive_rate":float(y.mean()),"rows":len(y),"group_rows":counts,"record_groups":{rid:assignment[rid] for rid in ids},"metrics":{}}
    tr,te=groups=="train",groups=="test"
    prior=float(y[tr].mean()); results["trivial_prior"]={"brier":float(brier_score_loss(y[te],np.full(te.sum(),prior))),"log_loss":float(log_loss(y[te],np.full(te.sum(),prior),labels=[0,1]))}
    for name,X in feats.items(): results["metrics"][name]=fit_eval(X[tr],y[tr],X[te],y[te])
    results["ordered_minus_unordered"]={k:results["metrics"]["C_ordered"][k]-results["metrics"]["B_unordered"][k] for k in ["brier","log_loss","auc"]}
    (out/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
    pd.DataFrame({"record":rec,"split":groups,"y":y}).to_csv(out/"target_rows.csv",index=False)
    print(json.dumps(results,indent=2))
if __name__=="__main__": main()
