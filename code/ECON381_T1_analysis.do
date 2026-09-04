*=============================================================*
* ECON 381-2 — Settlement-cycle compression & market quality
* T+1 (US, 2024-05-28) DiD pipeline   [Stata]
* SCOPE: uses ONLY commands practiced in 381-1/381-2 PSets:
*   xtset, xtreg ..., fe vce(cluster ...), reg ... robust, test,
*   i.dummies, ivregress 2sls, probit/logit.
*   (No reghdfe, no boottest/wild bootstrap, no -synth- : not taught.)
* Input : t1_panel.csv  (from ECON381_T1_data_pull.py)
* Prepared with Claude (Cowork), May 2026
*=============================================================*
clear all
set more off

*--- 0. Load & set up (cf. PS3) --------------------------------------*
import delimited "t1_panel.csv", clear varnames(1) case(lower)
gen sdate = date(date, "YMD")
format sdate %td
encode ticker, gen(lineid)          // each listing (US line OR home line) = panel entity
encode firm,   gen(firmid)          // company id (for clustering)
xtset lineid sdate

* outcomes (logs where sensible)
gen ln_spread   = ln(cs_spread + 1e-6)
gen ln_amihud   = ln(amihud)
gen ln_vol      = ln(parkinson + 1e-9)
gen ln_turnover = ln(turnover + 1)
local Y ln_spread ln_amihud ln_vol ln_turnover

*--- 1. Pre-period balance (reg + robust, cf. PS3) -------------------*
foreach v of local Y {
    di _n "Balance (pre-period): `v'"
    reg `v' treat if post==0, vce(cluster firmid)
}

*--- 2. Main DiD: two-way FE (entity FE + date dummies), cf. PS3/PS4 -*
* treatpost = treat*post is the DiD coefficient.
foreach v of local Y {
    di _n "==== DiD: `v' ===="
    xtreg `v' treatpost i.sdate, fe vce(cluster firmid)
}
* (Optional within-firm-day version, still in-scope = OLS with dummies:
*  egen firmdate = group(firm sdate)
*  reg ln_spread treatpost i.lineid i.firmdate, vce(cluster firmid)  )

*--- 3. Event study = DiD with time-dummy interactions (cf. PS3) ------*
* Build weekly event time relative to 2024-05-28; omit k=-1 as base.
gen relweek = floor(event_day/7)
keep if inrange(relweek,-12,12)
* treat x relweek interactions + entity FE + date dummies:
xtreg ln_spread ib(-1).relweek#i.treat i.sdate, fe vce(cluster firmid)
* joint test that all post-event interactions = 0 (cf. PS3 'test'):
testparm i(0/12).relweek#i.treat
* To graph, install once: ssc install coefplot
* coefplot, keep(*.relweek#1.treat) vertical yline(0) ///
*     xtitle("Weeks rel. to 2024-05-28") ytitle("DiD coef (log spread)")
* graph export "eventstudy_spread.png", replace width(1600)

*--- 4. Placebo: fake event 8 weeks early, pre-period only -----------*
preserve
keep if post==0
gen pl_post = sdate >= td(02apr2024)
gen pl_tp   = treat*pl_post
xtreg ln_spread pl_tp i.sdate, fe vce(cluster firmid)   // expect ~0
restore

*--- 5. (US) Binary outcome on settlement fails (cf. PS1) ------------*
* After merging SEC fails-to-deliver: gen fail = (ftd_shares>0)
* probit fail treatpost i.sdate, vce(cluster firmid)

*--- 6. Synthetic control (Lec 13): US market vs donor T+2 markets ----*
* Operates on a MARKET-level panel (one row per market-date): build an
* aggregate outcome (e.g., mean log spread, or a liquidity index) for the
* US and for each non-banning T+2 market (UK, DE, FR, JP, AU, CH, ...).
* ssc install synth
* use market_panel.dta, clear          // vars: marketid (US=1) date y
* tsset marketid date
* synth y y(2024w5) y(2024w10) y(2024w15) y,  ///   // pre-period predictors
*     trunit(1) trperiod(`=td(28may2024)') fig keep(synth_us.dta) replace
* graph export "synth_us.png", replace
* * post-event gap (treated - synthetic) = the estimated effect.

*--- 7. (India extension) Fuzzy DiD via IV (cf. PS2) -----------------*
* In the India do-file, eligibility Z instruments actual T+0 trading D:
* ivregress 2sls Y (D = Z) i.sdate, vce(cluster firmid)
* estat firststage     // report first-stage F

di _n "Done. Methods (all in 381 scope): panel FE DiD (PS3), event study via"
di    "dummies (PS3), synthetic control (Lec 13), IV/2SLS (PS2), probit (PS1)."
