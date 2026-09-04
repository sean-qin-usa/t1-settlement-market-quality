"""
ECON 381-2 — T+1 project: PIPELINE DRY-RUN ON *SIMULATED* DATA  (numpy-only)
===========================================================================
Proves the estimation pipeline runs and shows table/figure FORMAT.
NUMBERS ARE SIMULATED, NOT REAL FINDINGS. Real results = Stata .do on real CSV.
Outputs: did_table_SIM.txt, eventstudy_SIM.png, synth_SIM.png
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(381)
EVENT = pd.Timestamp("2024-05-28")

def ols_cluster(y, X, groups):
    """OLS with cluster-robust SEs (sandwich). X includes intercept column."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        Xg = X[groups == g]; ug = u[groups == g]
        sg = Xg.T @ ug
        meat += np.outer(sg, sg)
    V = XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V))

# ---- simulate within-firm cross-listing panel ---------------------------
firms = [f"F{i:02d}" for i in range(20)]
dates = pd.bdate_range("2024-01-02", "2024-09-30")
TRUE = -0.15
rows = []
for f in firms:
    ffe = rng.normal(0, .4)
    for treated in (1, 0):
        lfe = ffe + rng.normal(0, .2) + (.1 if treated else 0)
        for d in dates:
            post = int(d >= EVENT)
            y = -3.0 + lfe + .0008*(d-dates[0]).days + TRUE*treated*post + rng.normal(0,.25)
            rows.append((f, treated, d, y, post))
df = pd.DataFrame(rows, columns=["firm","treat","date","ln_spread","post"])
df["treatpost"] = df.treat*df.post
df["ticker"] = df.firm + np.where(df.treat==1,"_US","_HM")

# ---- 1. DiD: two-way FE (ticker + date dummies), cluster by firm --------
def design(frame, extra):
    parts = [np.ones((len(frame),1)), frame[extra].to_numpy()]
    parts.append(pd.get_dummies(frame["ticker"], drop_first=True, dtype=float).to_numpy())
    parts.append(pd.get_dummies(frame["date"], drop_first=True, dtype=float).to_numpy())
    return np.hstack(parts)

X = design(df, ["treatpost"]); y = df["ln_spread"].to_numpy()
beta, se = ols_cluster(y, X, df["firm"].to_numpy())
b, s = beta[1], se[1]; t = b/s
with open("did_table_SIM.txt","w") as fh:
    fh.write("ILLUSTRATIVE / SIMULATED DATA - NOT REAL RESULTS\n")
    fh.write("DiD: effect of T+1 on log bid-ask spread (US line vs home line)\n")
    fh.write(f"  treatpost = {b:+.4f}  cluster-SE = {s:.4f}  t = {t:+.2f}\n")
    fh.write(f"  (true simulated effect = {TRUE:+.2f})\n")
print(open("did_table_SIM.txt").read())

# ---- 2. Event study: treat x relweek dummies (omit k=-1) ---------------
df["relweek"] = ((df["date"]-EVENT).dt.days//7).clip(-12,12)
es = df[df.relweek!=-1].copy()
es["iw"] = es.treat.astype(str)+"_"+es.relweek.astype(int).astype(str)
Xe = np.hstack([np.ones((len(es),1)),
                pd.get_dummies(es["iw"], drop_first=True, dtype=float).to_numpy(),
                pd.get_dummies(es["ticker"], drop_first=True, dtype=float).to_numpy(),
                pd.get_dummies(es["date"], drop_first=True, dtype=float).to_numpy()])
cols = ["const"]+list(pd.get_dummies(es["iw"], drop_first=True).columns)
be, see = ols_cluster(es["ln_spread"].to_numpy(), Xe, es["firm"].to_numpy())
ix = {c:i for i,c in enumerate(cols)}
ks,co,lo,hi=[],[],[],[]
for k in range(-12,13):
    key=f"1_{k}"
    if key in ix:
        i=ix[key]; ks.append(k); co.append(be[i]); lo.append(be[i]-1.96*see[i]); hi.append(be[i]+1.96*see[i])
plt.figure(figsize=(8,4.5)); plt.axhline(0,color="grey",lw=.8); plt.axvline(-0.5,color="red",ls="--",lw=.8)
plt.errorbar(ks,co,yerr=[np.array(co)-np.array(lo),np.array(hi)-np.array(co)],fmt="o-",capsize=3)
plt.title("Event study (SIMULATED): T+1 effect on log spread")
plt.xlabel("Weeks relative to 2024-05-28"); plt.ylabel("DiD coefficient")
plt.figtext(0.5,0.005,"ILLUSTRATIVE / SIMULATED DATA - NOT REAL RESULTS",ha="center",color="red",fontsize=8)
plt.tight_layout(); plt.savefig("eventstudy_SIM.png",dpi=150); plt.close()

# ---- 3. Synthetic control (Lec 13): US market vs donor T+2 markets ------
T=len(dates); t0=(dates<EVENT).sum()
Dm=np.column_stack([np.cumsum(rng.normal(0,1,T))*0.5+rng.normal(0,1) for _ in range(6)])
us=Dm@np.array([.4,.3,.1,.1,.05,.05])+rng.normal(0,0.3,T); us[t0:]+=-0.8
A=np.column_stack([Dm[:t0],np.ones(t0)]); w=np.linalg.lstsq(A,us[:t0],rcond=None)[0][:-1]
w=np.clip(w,0,None); w=w/w.sum(); synth=Dm@w
plt.figure(figsize=(8,4.5)); plt.plot(dates,us,label="US market (treated)")
plt.plot(dates,synth,"--",label="Synthetic US (donor T+2 markets)")
plt.axvline(EVENT,color="red",ls=":",lw=.9); plt.legend()
plt.title("Synthetic control (SIMULATED): US vs synthetic-T+2")
plt.figtext(0.5,0.005,"ILLUSTRATIVE / SIMULATED DATA - NOT REAL RESULTS",ha="center",color="red",fontsize=8)
plt.tight_layout(); plt.savefig("synth_SIM.png",dpi=150); plt.close()
print("Saved did_table_SIM.txt, eventstudy_SIM.png, synth_SIM.png")
