#!/usr/bin/env python3
"""Causal, sample-indexed BIDMC recurrence pilot (keeps v1 untouched)."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ASSIGN = {"01":"train","02":"train","04":"train","05":"train","06":"train","10":"train","11":"train","12":"train","13":"train","14":"train","15":"train","16":"train","17":"train","19":"train","20":"val","21":"val","22":"val","23":"val","26":"val","27":"test","28":"test","29":"test","30":"test","32":"test"}
FS = 125.0

def causal_peaks(x, threshold):
    b, a = butter(3, [0.5/(FS/2), 8/(FS/2)], btype="band")
    z = lfilter(b, a, x)
    peaks=[]; last=-10**9; d=int(.30*FS)
    # A peak at j-1 is confirmed only when sample j has arrived.
    for j in range(2, len(z)):
        k=j-1
        if k-last < d: continue
        if z[k] >= z[k-1] and z[k] > z[j] and z[k] > threshold:
            peaks.append(k); last=k
    return np.asarray(peaks,dtype=int), z

def target_value(future):
    d=np.diff(future)
    a=d[:3]-np.mean(d[:3]); b=d[4:7]-np.mean(d[4:7])
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    if not np.isfinite(na+nb): return np.nan,"nonfinite"
    if na==0 or nb==0: return np.nan,"zero_norm"
    return float(np.dot(a,b)/(na*nb)),"valid"

def load(path, threshold, include_original=False):
    d=pd.read_csv(path); d.columns=[str(c).strip() for c in d.columns]
    if len(d)<50000 or "PLETH" not in d: return None
    x=d.PLETH.to_numpy(float); peaks,z=causal_peaks(x,threshold)
    rr=np.diff(peaks)/FS
    valid=(rr>=.30)&(rr<=2.0)
    # Invalid intervals are retained as gaps only for the audit; rows use contiguous valid runs.
    rows=[]
    for k in range(8,len(rr)-8):
        if not np.all(valid[k-8:k+8]):
            rows.append(dict(record=path.stem.split("_")[1],origin=int(peaks[k]+1),target=np.nan,valid=False,reason="invalid_interval",y_orig=np.nan))
            continue
        past=rr[k-8:k]; future=rr[k:k+8]; y,reason=target_value(future)
        d1,d2=future[1]-future[0],future[2]-future[1]
        yo=(int(d1*d2>0) if min(abs(d1),abs(d2))>=.015 and d1*d2!=0 else np.nan)
        state=np.array([np.mean(past[-4:]),past[-1],past[-1]-past[-2],np.median(np.abs(past-np.median(past)))])
        rows.append(dict(record=path.stem.split("_")[1],origin=int(peaks[k]+1),target=y,valid=np.isfinite(y),reason=reason,y_orig=yo, state=state, past=past))
    return dict(record=path.stem.split("_")[1],samples=len(x),peaks=len(peaks),rr_count=len(rr),valid_rr_frac=float(np.mean(valid)) if len(valid) else 0.,rows=rows)

def metrics(y,p):
    return {"mse":float(mean_squared_error(y,p)),"mae":float(mean_absolute_error(y,p)),"spearman":float(spearmanr(y,p).statistic) if len(np.unique(p))>1 and len(np.unique(y))>1 else None,"n":int(len(y))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--out",required=True); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    # Fit a single detector amplitude threshold from training recordings only.
    train_stds=[]; paths=sorted(Path(a.data).glob("bidmc_*_Signals.csv"))
    for p in paths:
        rid=p.stem.split("_")[1]
        if ASSIGN.get(rid)!="train": continue
        try:
            d=pd.read_csv(p); d.columns=[str(c).strip() for c in d.columns]; x=d.PLETH.to_numpy(float); _,z=causal_peaks(x,0.0); s=np.std(z[1000:]);
            if np.isfinite(s): train_stds.append(s)
        except Exception: pass
    if not train_stds: raise SystemExit("No training recordings for detector calibration")
    threshold=float(.15*np.median(train_stds)); audits=[]; rows=[]
    for p in paths:
        try: r=load(p,threshold)
        except Exception as e: print("skip",p,e); continue
        if r is None: continue
        audits.append({k:r[k] for k in ["record","samples","peaks","rr_count","valid_rr_frac"]})
        if ASSIGN.get(r["record"]) is None: continue
        for q in r["rows"]:
            q["split"]=ASSIGN[q["record"]]; rows.append(q)
    all_df=pd.DataFrame(rows); all_df.to_pickle(out/"internal_rows.pkl")
    # Only rows with a defined primary target enter identical model comparisons.
    ok=all_df.valid.fillna(False) & all_df.target.notna(); df=all_df[ok].copy();
    if len(df)<100 or df.split.eq("test").sum()<20: raise SystemExit("Insufficient valid target rows")
    states=np.stack(df.state); past=np.stack(df.past); un=np.sort(past,axis=1); order=past
    Xs={"A_state":states,"B_unordered":np.c_[states,un],"C_ordered":np.c_[states,order]}; y=df.target.to_numpy(float); tr=df.split.eq("train").to_numpy(); te=df.split.eq("test").to_numpy()
    preds=pd.DataFrame({"record":df.record,"split":df.split,"origin_sample":df.origin,"target":y,"target_valid":True})
    result={"fs_hz":FS,"detector_threshold":threshold,"audited_records":len(audits),"modeled_records":sorted(df.record.unique(),key=int),"record_assignment":ASSIGN,"rows_total":len(all_df),"target_valid_rows":int(ok.sum()),"target_coverage":float(ok.mean()),"target_missing_reasons":all_df.loc[~ok,"reason"].value_counts().to_dict(),"target_distribution":{"mean":float(np.mean(y)),"std":float(np.std(y)),"quantiles":{str(q):float(np.quantile(y,q)) for q in [.01,.25,.5,.75,.99]}},"target_definition":"centered cosine similarity between future change vectors d[0:3] and d[4:7], where d=np.diff(8 future intervals); d[3] separates patterns","models":{},"per_record":{}}
    # Detector perturbation audit: compare target values at the same confirmed origins.
    perturb={}
    base={(r,o):v for r,o,v in zip(df.record,df.origin,df.target)}
    for mult in [.9,1.1]:
        vals=[]; ref=[]
        for p in paths:
            rid=p.stem.split("_")[1]
            if rid not in df.record.unique(): continue
            try: rr=load(p,threshold*mult)
            except Exception: continue
            for q in rr["rows"]:
                key=(q["record"],q["origin"])
                if key in base and q["valid"] and np.isfinite(q["target"]): ref.append(base[key]); vals.append(q["target"])
        perturb[str(mult)]={"matched":len(vals),"fraction_of_valid":float(len(vals)/len(df)) if len(df) else 0.,"spearman":float(spearmanr(ref,vals).statistic) if len(vals)>2 and len(np.unique(vals))>1 else None,"mae":float(mean_absolute_error(ref,vals)) if vals else None}
    result["detector_perturbation"] = perturb
    train_mean=float(y[tr].mean()); result["baseline"]={"train_mean":train_mean,"test":metrics(y[te],np.full(te.sum(),train_mean))}
    result["target_state_associations"]={k:(float(spearmanr(y,states[:,i]).statistic) if len(np.unique(states[:,i]))>1 else None) for i,k in enumerate(["recent_level","latest_interval","recent_trend","past_mad"])}
    for name,X in Xs.items():
        sc=StandardScaler().fit(X[tr]); models={"linear_ridge":Ridge(alpha=1.0),"nonlinear_histgb":HistGradientBoostingRegressor(max_depth=2,learning_rate=.05,max_iter=100,min_samples_leaf=30,random_state=0)}
        for mn,m in models.items():
            m.fit(sc.transform(X[tr]),y[tr]); p=m.predict(sc.transform(X[te])); preds.loc[te,f"{name}_{mn}"]=p; result["models"].setdefault(mn,{})[name]=metrics(y[te],p)
    # Secondary corrected-preprocessing reproduction of v1's binary target.
    od=df.y_orig.notna().to_numpy(); odtr=od&tr; odte=od&te; result["original_target_diagnostic"]={"eligible_rows":int(od.sum()),"test_rows":int(odte.sum()),"future_conditioned":True,"metrics":{}}
    if odte.sum()>0 and len(np.unique(df.loc[od,"y_orig"]))==2:
        for name,X in Xs.items():
            sc=StandardScaler().fit(X[odtr]); m=LogisticRegression(C=1.0,max_iter=1000).fit(sc.transform(X[odtr]),df.loc[odtr,"y_orig"].to_numpy()); p=m.predict_proba(sc.transform(X[odte]))[:,1]; yy=df.loc[odte,"y_orig"].to_numpy(); result["original_target_diagnostic"]["metrics"][name]={"brier":float(brier_score_loss(yy,p)),"log_loss":float(log_loss(yy,p,labels=[0,1])),"auc":float(roc_auc_score(yy,p)) if len(np.unique(yy))==2 else None,"n":int(len(yy))}
    for rid,g in df[df.split.eq("test")].groupby("record"):
        result["per_record"][rid]={"n":int(len(g))}
        for mn in ["linear_ridge","nonlinear_histgb"]:
            for name in Xs: result["per_record"][rid][f"{name}_{mn}"]=metrics(g.target.to_numpy(),preds.loc[g.index,f"{name}_{mn}"].to_numpy())
    # Record bootstrap C-B for nonlinear MSE/MAE, test records only.
    rng=np.random.default_rng(0); ids=sorted(df.loc[te,"record"].unique(),key=int); diffs={"mse":[],"mae":[]}
    for _ in range(1000):
        pick=rng.choice(ids,len(ids),replace=True); ix=np.concatenate([df.index[(df.record==rid)&te] for rid in pick]); yy=df.loc[ix,"target"].to_numpy(); pb=preds.loc[ix,"B_unordered_nonlinear_histgb"].to_numpy(); pc=preds.loc[ix,"C_ordered_nonlinear_histgb"].to_numpy(); diffs["mse"].append(mean_squared_error(yy,pc)-mean_squared_error(yy,pb)); diffs["mae"].append(mean_absolute_error(yy,pc)-mean_absolute_error(yy,pb))
    result["record_bootstrap_C_minus_B"]={k:{"mean":float(np.mean(v)),"lo95":float(np.percentile(v,2.5)),"hi95":float(np.percentile(v,97.5))} for k,v in diffs.items()}
    pd.DataFrame(audits).to_csv(out/"record_audit.csv",index=False); preds.to_csv(out/"predictions.csv",index=False); pd.DataFrame([{**{"record":rid},**v} for rid,v in result["per_record"].items()]).to_json(out/"per_record_metrics.json",orient="records",indent=2); (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
