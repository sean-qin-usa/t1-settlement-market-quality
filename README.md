# Does Compressing the Settlement Cycle Improve Market Quality?

**Evidence from the 2024 US move to T+1 settlement.**

Final project for **ECON 381-2 (Econometrics)** at **Northwestern University** — Sean Qin, June 2026.

## Summary

On 28 May 2024 the US shortened the standard equity settlement cycle from T+2 to T+1 while European markets stayed on T+2. This paper estimates the effect on market quality with a within-firm difference-in-differences design: for 19 cross-listed firms (38 listings, 7,184 listing-days), the US-listed line is treated while the same firm's home-market line serves as the control.

Headline findings:

- **Transaction costs:** a bounded null. The spread coefficient is 0.010 (SE 0.289) — no evidence of an effect, though the confidence interval is too wide to call it a tight zero. The null survives weekly aggregation and reappears in an independent non-cross-listed US-vs-EU sample.
- **Volatility:** a large negative coefficient (−0.198) that fails a pre-period placebo and flips sign under an alternative control group — read as a differential trend, not a causal effect of T+1.
- **Methodological lessons:** an initial 8-pair sample produced a spurious "significant" spread effect (−1.05), an instance of the few-cluster inference problem; placebo and alternative-control tests are what separate causal estimates from trends.

## Repository layout

```
paper/   main.tex + figures (compiles with pdflatex; also lives on Overleaf)
code/    data pull, analysis, and demo scripts
data/    raw and constructed panels (daily OHLCV from Yahoo Finance, Jan–Sep 2024)
```

### Code

| File | What it does |
|---|---|
| `code/ECON381_T1_data_pull.py` | Pulls daily OHLCV via `yfinance` for the cross-listed pairs and builds the listing-day panel |
| `code/ECON381_T1_run.py` | Python estimation on `data/t1_panel_full.csv` — reproduces the Stata two-way FE DiD with firm-clustered SEs and generates the event-study figure |
| `code/ECON381_T1_analysis.do` | Stata pipeline (course-scope commands: `xtreg, fe vce(cluster)`, event-study dummies, placebo) |
| `code/ECON381_T1_sim_demo.py` | Pipeline dry-run on simulated data (format check — numbers are not real findings) |
| `code/ECON381_T1_selfcontained_DEMO.do` | Self-contained Stata demo on simulated data |

### Data

- `data/t1_raw.csv` — raw daily OHLCV by firm/ticker/date
- `data/t1_panel_full.csv` — full listing-day panel used in estimation

## Reproducing

```bash
pip install yfinance pandas numpy matplotlib
python code/ECON381_T1_data_pull.py   # rebuild the panel (or use data/ as-is)
python code/ECON381_T1_run.py         # main estimates + event-study figure
```

Analysis was prepared with the help of Claude (Anthropic).
