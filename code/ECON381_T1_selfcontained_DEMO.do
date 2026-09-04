*=============================================================*
* ECON 381-2 — T+1 project: SELF-CONTAINED Stata DEMO
* Paste this whole file into the Apporto Stata do-editor and run.
* It SIMULATES a realistic within-firm panel and runs the full
* in-scope pipeline so you get real Stata tables + an event-study
* graph to screenshot. Swap in the real t1_panel.csv when ready.
* Methods: panel-FE DiD (PS3), event study via dummies (PS3),
*          synthetic control (Lec 13). All in 381 scope.
* DATA HERE IS SIMULATED — illustrative output, not real findings.
*=============================================================*
clear all
set more off
set seed 381

*--- 0. Simulate a within-firm cross-listing panel -------------------*
* 20 firms x 2 listings (US treated / home control) x ~190 trading days
local NF = 20
set obs `NF'
gen firm = _n
gen firm_fe = rnormal(0,.4)
expand 2
bys firm: gen treat = (_n==1)          // 1 = US line (T+1), 0 = home line
gen line_fe = firm_fe + rnormal(0,.2) + .1*treat
expand 190
bys firm treat: gen t = _n              // trading-day index 1..190
gen sdate = td(02jan2024) + t           // calendar-ish
format sdate %td
gen post = t > 100                      // event at t=100 (~2024-05-28)
gen treatpost = treat*post
scalar TRUE = -0.15                     // planted effect: T+1 lowers spread 15%
gen ln_spread = -3 + line_fe + .002*t + TRUE*treatpost + rnormal(0,.25)
egen lineid = group(firm treat)
xtset lineid t

*--- 1. Main DiD: two-way FE, clustered by firm (cf. PS3) ------------*
di _n "==== DiD: effect of T+1 on log spread  (true = -0.15) ===="
xtreg ln_spread treatpost i.t, fe vce(cluster firm)
* the 'treatpost' coefficient is your ATT.

*--- 2. Event study = treat x event-week dummies (cf. PS3) -----------*
gen relweek = floor((t-100)/5)          // ~weekly bins around event
replace relweek = -12 if relweek < -12
replace relweek =  12 if relweek >  12
* run interactions (omit relweek = -1 as base), entity + time FE:
xtreg ln_spread ib(-1).relweek#i.treat i.t, fe vce(cluster firm)
testparm i(0/12).relweek#i.treat        // joint test of post effects (cf. PS3 'test')

* build the event-study graph natively (no add-ons):
tempname M
postfile `M' k b lo hi using es_coefs, replace
forvalues k = -12/12 {
    if `k' == -1 continue
    capture lincom 1.treat#`k'.relweek
    if _rc==0 post `M' (`k') (r(estimate)) (r(estimate)-1.96*r(se)) (r(estimate)+1.96*r(se))
}
postclose `M'
preserve
use es_coefs, clear
twoway (rcap hi lo k) (connected b k), yline(0) xline(-1) ///
    title("Event study (SIMULATED): T+1 effect on log spread") ///
    xtitle("Weeks relative to event") ytitle("DiD coefficient") legend(off)
graph export "eventstudy_stata.png", replace width(1600)
restore

*--- 3. Placebo: fake event in the pre-period (should be ~0) ---------*
preserve
keep if post==0
gen pl_post = t > 50
gen pl_tp = treat*pl_post
xtreg ln_spread pl_tp i.t, fe vce(cluster firm)
restore

*--- 4. Synthetic control (Lec 13) ----------------------------------*
* needs Abadie's -synth- (Apporto usually allows ssc):
capture ssc install synth
* build a tiny market-level panel: US (treated id=1) vs 6 donor T+2 markets
preserve
clear
set obs 7
gen mkt = _n
expand 190
bys mkt: gen t = _n
gen y = 5 + 0.01*t + rnormal(0,.3) + 0.5*mkt
replace y = y - 0.8 if mkt==1 & t>100      // treated US drops post-event
xtset mkt t
capture synth y y(1) y(40) y(80) y, trunit(1) trperiod(101) fig
capture graph export "synth_stata.png", replace width(1600)
restore

di _n "DONE. Tables in the Results window; graphs saved:"
di    "  eventstudy_stata.png , synth_stata.png"
di    "(SIMULATED data — replace with real t1_panel.csv for real numbers.)"
