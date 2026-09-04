"""
ECON 381-2 T+1 — REAL analysis on Yahoo data (t1_raw.csv).
Reproduces Stata: reg Y treatpost i.ticker i.date, vce(cluster firm)
(point estimate == xtreg, fe; cluster-robust SE w/ Stata small-sample q_c).
Outputs: console tables + eventstudy_REAL.png
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df = pd.read_csv("t1_panel_full.csv")
df = df[df.firm!="ERR"].copy()
# keep only complete pairs (both a treated US line and a control home line)
df = df[df.groupby("firm")["treat"].transform("nunique")==2]
for c in ["open","high","low","close","volume"]:
    df[c]=pd.to_numeric(df[c],errors="coerce")
df["date"]=pd.to_datetime(df["date"])
df=df.dropna(subset=["open","high","low","close","volume"])
df=df.sort_values(["ticker","date"])

# ---- liquidity / volatility proxies (daily; data construction) ----
g=df.groupby("ticker",group_keys=False)
df["ret"]=g["close"].pct_change()
df["dvol"]=df["close"]*df["volume"]
df["amihud"]=(df["ret"].abs()/df["dvol"]).replace([np.inf,-np.inf],np.nan)*1e6
hl=np.log(df["high"]/df["low"]); df["parkinson"]=np.sqrt(hl**2/(4*np.log(2)))
def cs(x):
    h,l=x["high"].values,x["low"].values; n=len(x); out=np.full(n,np.nan); k=1/(3-2*np.sqrt(2))
    for t in range(1,n):
        hi2,lo2=max(h[t],h[t-1]),min(l[t],l[t-1])
        if min(l[t],l[t-1],lo2)<=0: continue
        beta=np.log(h[t]/l[t])**2+np.log(h[t-1]/l[t-1])**2; gamma=np.log(hi2/lo2)**2
        a=(np.sqrt(2*beta)-np.sqrt(beta))*k-np.sqrt(gamma*k); out[t]=max(2*(np.exp(a)-1)/(1+np.exp(a)),0)
    return pd.Series(out,index=x.index)
df["cs_spread"]=g.apply(cs).reset_index(level=0,drop=True)
df["turnover"]=df["volume"]

EVENT=pd.Timestamp("2024-05-28")
df["treat"]=df["treat"].astype(int)
df["post"]=(df["date"]>=EVENT).astype(int)
df["treatpost"]=df["treat"]*df["post"]
df["ln_spread"]=np.log(df["cs_spread"]+1e-6)
df["ln_amihud"]=np.log(df["amihud"].clip(lower=1e-9))
df["ln_vol"]=np.log(df["parkinson"]+1e-9)
df["ln_turn"]=np.log(df["turnover"]+1)

def ols_cluster(y,X,cl):
    XtX_inv=np.linalg.pinv(X.T@X); b=XtX_inv@(X.T@y); u=y-X@b
    K=np.linalg.matrix_rank(X); N=len(y); G=len(np.unique(cl))
    meat=np.zeros((X.shape[1],)*2)
    for gg in np.unique(cl):
        Xg=X[cl==gg]; ug=u[cl==gg]; s=Xg.T@ug; meat+=np.outer(s,s)
    qc=(N-1)/(N-K)*G/(G-1)            # Stata's small-sample correction
    V=qc*(XtX_inv@meat@XtX_inv)
    return b,np.sqrt(np.diag(V)),N,G

def design(frame,xcols):
    parts=[np.ones((len(frame),1)),frame[xcols].to_numpy(float)]
    parts.append(pd.get_dummies(frame["ticker"],drop_first=True,dtype=float).to_numpy())
    parts.append(pd.get_dummies(frame["date"].dt.strftime("%Y%m%d"),drop_first=True,dtype=float).to_numpy())
    return np.hstack(parts)

# ---- descriptive statistics (pre-period), for the paper's Table 1 ----
pre=df[df.post==0]
desc=pre.groupby("treat")[["cs_spread","amihud","parkinson","ret","volume"]].agg(["mean","std","median"]).T
desc.to_csv("descriptive_stats.csv")
print("DESCRIPTIVE STATS (pre-period, by treated):")
print(pre.groupby("treat")[["cs_spread","parkinson","ret"]].mean().round(5))
print("obs:",len(df),"firms:",df.firm.nunique(),"listings:",df.ticker.nunique())

print("="*64)
print("REAL DiD: effect of US T+1 (28 May 2024) on market quality")
print("Spec: reg Y treatpost i.ticker i.date, vce(cluster firm)  | clusters(firm)=8")
print("="*64)
print(f"{'Outcome':14}{'treatpost':>12}{'clusterSE':>12}{'t':>8}{'N':>7}")
res={}
for y,lab in [("ln_spread","ln spread"),("ln_amihud","ln Amihud"),("ln_vol","ln volatility"),("ln_turn","ln turnover")]:
    d=df.dropna(subset=[y]).copy()
    X=design(d,["treatpost"]); b,se,N,G=ols_cluster(d[y].to_numpy(),X,d["firm"].to_numpy())
    print(f"{lab:14}{b[1]:12.4f}{se[1]:12.4f}{b[1]/se[1]:8.2f}{N:7d}")
    res[y]=(b[1],se[1])

# ---- event study on ln_spread ----
d=df.dropna(subset=["ln_spread"]).copy()
d["relw"]=((d["date"]-EVENT).dt.days//7).clip(-12,12)
d=d[d["relw"]!=-1].copy()
d["iw"]=d["treat"].astype(str)+"_"+d["relw"].astype(int).astype(str)
X=np.hstack([np.ones((len(d),1)),
    pd.get_dummies(d["iw"],drop_first=True,dtype=float).to_numpy(),
    pd.get_dummies(d["ticker"],drop_first=True,dtype=float).to_numpy(),
    pd.get_dummies(d["date"].dt.strftime("%Y%m%d"),drop_first=True,dtype=float).to_numpy()])
cols=["c"]+list(pd.get_dummies(d["iw"],drop_first=True).columns)
b,se,N,G=ols_cluster(d["ln_spread"].to_numpy(),X,d["firm"].to_numpy())
ix={c:i for i,c in enumerate(cols)}
ks,co,lo,hi=[],[],[],[]
for k in range(-12,13):
    key=f"1_{k}"
    if key in ix:
        i=ix[key]; ks.append(k); co.append(b[i]); lo.append(b[i]-1.96*se[i]); hi.append(b[i]+1.96*se[i])
plt.figure(figsize=(8,4.5)); plt.axhline(0,color="grey",lw=.8); plt.axvline(-0.5,color="red",ls="--",lw=.8)
plt.errorbar(ks,co,yerr=[np.array(co)-np.array(lo),np.array(hi)-np.array(co)],fmt="o-",capsize=3)
plt.title("Event study (REAL data): US T+1 effect on log spread")
plt.xlabel("Weeks relative to 2024-05-28"); plt.ylabel("DiD coefficient (treated x week)")
plt.tight_layout(); plt.savefig("eventstudy_REAL.png",dpi=150)
print("\nSaved eventstudy_REAL.png  | firms:",df.firm.nunique()," listings:",df.ticker.nunique())
