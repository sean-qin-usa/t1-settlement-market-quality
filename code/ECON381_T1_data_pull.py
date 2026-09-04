"""
ECON 381-2 — T+1 / T+0 settlement project: DATA PULLER
======================================================
RUN ON YOUR OWN MACHINE (the Cowork sandbox can't reach Yahoo).
Pulls free daily OHLCV, computes liquidity/volatility proxies, and writes a
firm-date panel CSV that the Stata do-file (ECON381_T1_analysis.do) consumes.

    pip install yfinance pandas numpy
    python ECON381_T1_data_pull.py      ->  t1_panel.csv

Design: US-listed lines (treated, T+1 from 2024-05-28) vs same firm's home
T+2 line / ADR pair (control). Edit the PAIRS list with real cross-listed names.
Prepared with Claude (Cowork), May 2026.
"""
import numpy as np, pandas as pd

US_EVENT = "2024-05-28"          # US/Canada/Mexico -> T+1
WINDOW   = ("2024-01-02", "2024-09-30")

# Cross-listed pairs: (US_ticker [treated, T+1], HOME_ticker [control, T+2], firm)
# US line is the ADR/US listing; HOME line trades on a T+2 market (LSE .L, Euronext .PA/.AS, etc.)
PAIRS = [
    ("UL",   "ULVR.L",  "Unilever"),
    ("BP",   "BP.L",    "BP"),
    ("RIO",  "RIO.L",   "Rio Tinto"),
    ("SAP",  "SAP.DE",  "SAP"),
    ("TTE",  "TTE.PA",  "TotalEnergies"),
    ("NVS",  "NOVN.SW", "Novartis"),
    ("AZN",  "AZN.L",   "AstraZeneca"),
    ("SHEL", "SHEL.L",  "Shell"),
    # TODO: expand; verify each home line is on a T+2 market in 2024.
]

def fetch(tk, start, end):
    import yfinance as yf
    df = yf.download(tk, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        print("  WARN no data:", tk); return None
    df = df.rename(columns=str.lower)[["open","high","low","close","volume"]].reset_index()
    df.columns = ["date","open","high","low","close","volume"]
    df["ticker"] = tk
    return df

def proxies(df):
    df = df.sort_values("date").copy()
    df["ret"] = df["close"].pct_change()
    df["dvol"] = df["close"]*df["volume"]
    df["amihud"] = (df["ret"].abs()/df["dvol"]).replace([np.inf,-np.inf],np.nan)*1e6
    hl = np.log(df["high"]/df["low"]); df["parkinson"] = np.sqrt(hl**2/(4*np.log(2)))
    df["cs_spread"] = corwin_schultz(df)
    df["turnover"] = df["volume"]
    return df

def corwin_schultz(x):
    h,l = x["high"].values, x["low"].values; n=len(x); out=np.full(n,np.nan)
    k = 1/(3-2*np.sqrt(2))
    for t in range(1,n):
        hi2,lo2 = max(h[t],h[t-1]), min(l[t],l[t-1])
        if min(l[t],l[t-1],lo2) <= 0: continue
        beta = np.log(h[t]/l[t])**2 + np.log(h[t-1]/l[t-1])**2
        gamma = np.log(hi2/lo2)**2
        alpha = (np.sqrt(2*beta)-np.sqrt(beta))*k - np.sqrt(gamma*k)
        out[t] = max(2*(np.exp(alpha)-1)/(1+np.exp(alpha)), 0.0)
    return pd.Series(out, index=x.index)

rows = []
for us, home, firm in PAIRS:
    for tk, treated in [(us,1),(home,0)]:
        d = fetch(tk, *WINDOW)
        if d is None: continue
        d = proxies(d)
        d["firm"], d["treat"] = firm, treated
        rows.append(d)

panel = pd.concat(rows, ignore_index=True)
ev = pd.Timestamp(US_EVENT)
panel["post"] = (panel["date"] >= ev).astype(int)
panel["treatpost"] = panel["treat"]*panel["post"]
panel["event_day"] = (panel["date"] - ev).dt.days        # calendar; do-file rebuilds trading-day k
panel.to_csv("t1_panel.csv", index=False)
print("Wrote t1_panel.csv  rows:", len(panel), " firms:", panel['firm'].nunique())
print("NEXT: open ECON381_T1_analysis.do in Stata and run.")
