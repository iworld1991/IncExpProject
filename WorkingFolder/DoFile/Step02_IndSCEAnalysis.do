clear
global mainfolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder"
global folder "${mainfolder}/SurveyData/"
global otherdata_folder "${mainfolder}/OtherData"
global otherfolder "/Users/Myworld/Dropbox/SearchMatchExpectation/"
global graph_folder "${mainfolder}/Graphs/sce/"
global sum_graph_folder "${mainfolder}/Graphs/ind"
global sum_table_folder "${mainfolder}/Tables"


cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/indSCE_log",replace


*********************************************************
*** before working with SCE, clean the macro monthly data 
********************************************************

use "${mainfolder}/OtherData/macroM_raw.dta",clear 

generate new_date = dofc(DATE)
*format new_date %tm
gen year = year(new_date)
gen month = month(new_date)
gen date_str = string(year)+ "m"+string(month)
gen date = monthly(date_str,"YM")
format date %tm
drop DATE date_str year month new_date 
label var sp500 "growth rate (%) of sp500 index from last month"
rename UNRATE uerate
label var uerate "unemployment rate"
label var he "quarterly growth in hourly earning"
rename VIXCLS vix 
label var vix "vix indices"
save "${mainfolder}/OtherData/macroM.dta",replace 
clear 


*****************************************
** Prepare estimated income risks data **
******************************************

** full sample
use "${otherdata_folder}/sipp/sipp_history_vol_decomposed.dta", clear 

drop index 
gen year = floor(YM/100)
gen month = YM - year*100
gen date_str=string(year)+"M"+string(month)
gen date = monthly(date_str,"YM")
format date %tm
drop date_str 

table date

tsset date
tsfill
replace month=1 if f1.month==2 & month==.
replace year = f1.year if f1.month==2 & month==1
replace YM = 100*year+month if YM==.


rename permanent prisk_all
rename transitory trisk_all
label var prisk_all "estimated full-sample permanent risk(std)"
label var trisk_all "estimated full-sample transitory risk (std)"

** estimated risks in variances 

gen prisk2_all=prisk_all^2
gen trisk2_all=trisk_all^2


label var prisk2_all "full sample permanent risk (var)"
label var trisk2_all "full sample transitory risk  (var)"


***********************************
** Fill some estimation risk for january 
*************************************

replace prisk2_all=(l1.prisk2_all+f1.prisk2_all)/2 if month==1
replace trisk2_all=(l1.trisk2_all+f1.trisk2_all)/2 if month==1

***********************************
** Time aggregation of estimation risk
*************************************

foreach x in all{
gen prisk2_`x'_rl = (f1.prisk2_`x'+f2.prisk2_`x'+f3.prisk2_`x'+f4.prisk2_`x'+ ///
                      f5.prisk2_`x'+f6.prisk2_`x'+f7.prisk2_`x'+f8.prisk2_`x'+ ///
					  f9.prisk2_`x'+ f10.prisk2_`x'+ f11.prisk2_`x'+f12.prisk2_`x') 

gen trisk2_`x'_rl = (f1.trisk2_`x'+f2.trisk2_`x'+f3.trisk2_`x'+f4.trisk2_`x'+ ///
                      f5.trisk2_`x'+f6.trisk2_`x'+f7.trisk2_`x'+f8.trisk2_`x'+ ///
					  f9.trisk2_`x'+ f10.trisk2_`x'+ f11.trisk2_`x'+f12.trisk2_`x')/12 

gen rincvar_`x'_rl = prisk2_`x'_rl+ trisk2_`x'_rl
}

label var rincvar_all_rl "realized full-sample annual risk 12 m from now"


gen rincvar_all_now = l12.rincvar_all_rl
label var rincvar_all_now "realized full-sample annual risk since 12 m ago"

drop year month
save "${otherdata_folder}/sipp/sipp_history_vol_decomposed_annual.dta",replace 


*****************************
** subsample sample
******************************

use "${otherdata_folder}/sipp/sipp_history_vol_decomposed_edu_gender_age5.dta", clear

drop index 
gen year = floor(YM/100)
gen month = YM - year*100
gen date_str=string(year)+"M"+string(month)
gen date = monthly(date_str,"YM")
format date %tm
drop date_str 

table date
************************************************
egen groupid = group(gender educ age_5yr)
xtset groupid date 
********************************************
tsfill
replace month=1 if f1.month==2 & month==.
replace year = f1.year if f1.month==2 & month==1
replace YM = 100*year+month if YM==.

rename permanent prisk_sub
rename transitory trisk_sub
label var prisk_sub "estimated sub-sample permanent risk(std)"
label var trisk_sub "estimated sub-sample transitory risk (std)"

** estimated risks in variances 
gen prisk2_sub=prisk_sub^2
gen trisk2_sub = trisk_sub^2 

label var prisk2_sub "sub sample permanent risk  (var)"
label var trisk2_sub "sub sample transitory risk  (var)"


foreach var in prisk2_sub trisk2_sub{
      egen `var'pl=pctile(`var'),p(1) by(date)
	  egen `var'pu=pctile(`var'),p(99) by(date)
	  replace `var' = . if `var' <`var'pl | (`var' >`var'pu & `var'!=.)
}

***********************************
** Fill some estimation risk for january 
*************************************

replace prisk2_sub=(l1.prisk2_sub+f1.prisk2_sub)/2 if month==1
replace trisk2_sub=(l1.trisk2_sub+f1.trisk2_sub)/2 if month==1

***********************************
** Time aggregation of estimation risk
*************************************

foreach x in sub{
gen prisk2_`x'_rl = (f1.prisk2_`x'+f2.prisk2_`x'+f3.prisk2_`x'+f4.prisk2_`x'+ ///
                      f5.prisk2_`x'+f6.prisk2_`x'+f7.prisk2_`x'+f8.prisk2_`x'+ ///
					  f9.prisk2_`x'+ f10.prisk2_`x'+ f11.prisk2_`x'+f12.prisk2_`x') 

gen trisk2_`x'_rl = (f1.trisk2_`x'+f2.trisk2_`x'+f3.trisk2_`x'+f4.trisk2_`x'+ ///
                      f5.trisk2_`x'+f6.trisk2_`x'+f7.trisk2_`x'+f8.trisk2_`x'+ ///
					  f9.trisk2_`x'+ f10.trisk2_`x'+ f11.trisk2_`x'+f12.trisk2_`x')/12 

gen rincvar_`x'_rl = prisk2_`x'_rl+ trisk2_`x'_rl
}

label var rincvar_sub_rl "realized sub-sampel annual risk"

gen rincvar_sub_now = l12.rincvar_sub_rl
label var rincvar_sub_now "realized sub-sample annual risk since 12m ago"

drop year month groupid 
save "${otherdata_folder}/sipp/sipp_history_vol_decomposed_edu_gender_age5_annual.dta",replace
 
***************************
**  Clean and Merge Data **
***************************

use "${folder}/SCE/IncExpSCEProbIndM",clear 

duplicates report year month userid

************************************
**  merge with estimated moments **
***********************************

* IncExpSCEDstIndM is the output from ../Pythoncode/DoDensityEst.ipynb
merge 1:1 userid date using "${folder}/SCE/IncExpSCEDstIndM.dta", keep(master match) 
rename _merge IndEst_merge

duplicates report year month userid


** merge labor market module  
merge 1:1 year month userid using "${otherfolder}/data/SCE/LaborExpSCEIndM", keep(master match)
rename _merge labor_merge
rename userid ID 

*****************************************
****  Renaming so that more consistent **
*****************************************

rename Q24_mean incmean
rename Q24_var incvar
rename Q24_iqr inciqr
rename IncSkew incskew 
rename Q24_rmean rincmean
rename Q24_rvar rincvar

*rename Q36 educ
rename D6 HHinc 
rename Q32 age 
rename Q33 gender 
gen fulltime = Q10_1 
gen parttime = Q10_2 
rename Q12new selfemp
gen Stkprob= Q6new/100
gen UEprobAgg=Q4new/100
gen  UEprobInd4m = oo1_5/100
gen UEprobInd=Q13new/100
rename Q26v2 spending_dum
rename Q26v2part2 spending 
replace spending = spending/100

gen exp_offer = oo2u/100
gen EUprobInd = Q22new/100

***********************
** merge other data ***
**************************


**macro data
merge m:1 date using "${mainfolder}/OtherData/macroM.dta",keep(master match)
rename CPIAUCSL CPIAU
drop _merge 


* flow rate data
merge m:1 year month using "${otherfolder}/workingdata/CPS_worker_flows.dta",keep(master match)
drop _merge 


* income risk estimation data
gen YM = year*100+month

** merge population estimation 
merge m:1 YM using "${otherdata_folder}/sipp/sipp_history_vol_decomposed_annual.dta", keep(master match)
drop _merge 

********************
** group variables *
********************

egen byear_5yr = cut(byear), ///
     at(1915 1920 1925 1930 1935 1940 ///
	    1945 1950 1955 1960 1965 1970 ///
	    1975 1980 1985 1990 1995 2000 ///
		2005 2010)
label var byear_5yr "5-year cohort"

egen age_5yr = cut(age), ///
     at(20 25 30 35 40 45 ///
	    50 55 60)
label var age_5yr "5-year age"


gen edu_g = . 
replace edu_g = 1 if educ==1
replace edu_g = 2 if educ==2 | educ ==3 | educ == 4
replace edu_g = 3 if educ <=9 & educ>4

label var edu_g "education group"
label define edu_glb 1 "HS dropout" 2 "HS graduate" 3 "College/above"
label value edu_g edu_glb

replace educ = edu_g 

merge m:1 gender educ age_5yr YM ///
         using "${otherdata_folder}/sipp/sipp_history_vol_decomposed_edu_gender_age5_annual.dta", keep(master match)

drop _merge 

*******************************
**  Set Panel Data Structure **
*******************************

xtset ID date
sort ID date

*******************************
** Exclude youth and retired
******************************
keep if age <= 60 & age >= 20

*******************************
**  Summary Statistics of SCE **
*******************************

tabstat ID,s(count) by(date) column(statistics)

*******************************
**  Generate Variables       **
*******************************

gen Incmean = .
gen Incvar = .
gen Inciqr = .

gen Incmean_ch = .
gen Incvar_ch = .
gen Inciqr_ch = .


gen Inc_mom = .
gen Inc_mom_ch = .

*** take the log

foreach mom in var{
gen linc`mom'= log(inc`mom') 
gen lrinc`mom' = log(rinc`mom') 
}


foreach mom in iqr{
gen linc`mom'= log(inc`mom') 
}

label var linciqr "log perceived iqr"
label var lincvar "log perceived risk"
label var lrincvar "log perceived risk (real)"
label var spending "expected growth in spending"

*************************************
*** Poisson rate in expectations 
**************************************


gen exp_s = log(1-UEprobInd4m)/(-3) 
* can use exp_eu from micro data 
label var exp_s "expected Poisson separation rate"

gen exp_f =log(1-EUprobInd)/(-4)
* can use exp_ue from micro data 
label var exp_f "expected Poisson job-finding rate"

gen exp_s_1y = log(1-UEprobInd)/(-12) 
* can use exp_eu from micro data 
label var exp_s_1y "expected Poisson separation rate (1 year)"

*************************************
*** Variables for overreaction test 
**************************************


** earning risk 

gen rincvar_rv = rincvar - l1.rincvar
label var rincvar "revision in earning risk"
gen incvar_rv = incvar - l1.incvar
label var incvar "revision in earning risk"

gen rincvar_all_fe = rincvar-rincvar_all_rl
label var rincvar_all_fe "forecast error in earning risk"
gen rincvar_sub_fe = rincvar-rincvar_sub_rl
label var rincvar_all_fe "forecast error in earning risk"

** negative fe means underestimation 

foreach var in prisk2_all trisk2_all prisk2_sub trisk2_sub rincvar_all_now rincvar_sub_now{
gen `var'_ch = `var'-l1.`var'
}

** separation risk
gen exp_s_1y_fe = exp_s_1y - f1.s
label var exp_s_1y_fe "forecast error of separation rate"

gen exp_s_1y_rv = exp_s_1y - l1.exp_s_1y
label var exp_s_1y_rv "revision in separation rate"

gen exp_s_fe = exp_s - f4.s
label var exp_s_fe "forecast error of separation rate"
gen exp_s_rv = exp_s - l4.exp_s
label var exp_s_rv "revision in separation rate"

gen exp_f_fe = exp_f - f1.f
label var exp_s_fe "forecast error of separation rate"

gen exp_f_rv = exp_f - l1.exp_f
label var exp_f_rv "revision in separation rate"

***********************************************
**** Job experiences/transitions/search *****
***********************************************

gen emp_status = .

replace emp_status = 1 if Q10_1==1|Q10_2==1|Q10_5==1
replace emp_status = 2 if Q10_3==1|Q10_4==1
replace emp_status = 3 if Q10_6==1|Q10_7==1|Q10_8==1|Q10_9==1

label define emp_status_lb 1 "employed" 2 "unemployed" 3 "non-working"
label value emp_status emp_status_lb

gen e2u = cond(emp_status==2 & (l1.emp_status==1|l2.emp_status==1|l3.emp_status==1|l4.emp_status==1),1,0)
label var e2u "employed since m-4 and unemployed now"

gen u2e = cond(emp_status==1 &  (l1.emp_status==2|l2.emp_status==2|l3.emp_status==2|l4.emp_status==2),1,0)
label var u2e "unemployed since m-4 and employed now"


gen e2um8 = cond(emp_status==2 & ///
                 (l1.emp_status==1|l2.emp_status==1|l3.emp_status==1|l4.emp_status==1| ///
				  l5.emp_status==1|l6.emp_status==1|l7.emp_status==1|l8.emp_status==1),1,0)
label var e2um8 "employed since m-8 and unemployed now"

gen u2em8 = cond(emp_status==1 &  ///
               (l1.emp_status==2|l2.emp_status==2|l3.emp_status==2|l4.emp_status==2| ///
			   l5.emp_status==2|l6.emp_status==2|l7.emp_status==2|l5.emp_status==2),1,0)
label var u2em8 "unemployed since m-8 and employed now"


gen e2um12 = cond(emp_status==2 & ///
                 (l1.emp_status==1|l2.emp_status==1|l3.emp_status==1|l4.emp_status==1| ///
				  l5.emp_status==1|l6.emp_status==1|l7.emp_status==1|l8.emp_status==1| ///
				  l9.emp_status==1|l10.emp_status==1|l11.emp_status==1|l12.emp_status==1),1,0)
label var e2um12 "employed since y-1 and unemployed now"

gen u2em12 = cond(emp_status==1 &  ///
               (l1.emp_status==2|l2.emp_status==2|l3.emp_status==2|l4.emp_status==2| ///
			   l5.emp_status==2|l6.emp_status==2|l7.emp_status==2|l5.emp_status==2| ///
			    l9.emp_status==2|l10.emp_status==2|l11.emp_status==2|l12.emp_status==2),1,0)
label var u2em12 "unemployed since y-1 and employed now"

*************************
*** Exclude outliers *****
*************************

local vars incvar rincvar l3

foreach var in `vars'{
      egen `var'pl=pctile(`var'),p(1)
	  egen `var'pu=pctile(`var'),p(99)
	  replace `var' = . if `var' <`var'pl | (`var' >`var'pu & `var'!=.)
}

*************************
** macro variables normalization
*****************************

replace he = he*l3.CPIAU/CPIAU
label var he "real hourly earning growth"

gen ue_chg = uerate - l3.uerate
label var ue_chg "change in unemployment rate"

************************
** group variables ****
************************
** age square
gen age2 = age^2
label var age2 "age squared"


***********************************
** new variables SCE realizations **
**********************************

* log earning 
gen wage_1y = l3*100/CPIAU
label var wage_1y "average real annual earning from primary job"

** average earning 
egen wage_1y_av = mean(wage_1y), by(ID)
label var wage_1y_av "average real annual earning across time"

** income decile
egen earning_gp = xtile(wage_1y_av), by(date) n(10) 
label var earning_gp "decile of earning from low to high"

gen lwage_1y = log(wage_1y)
label var lwage_1y "log average annual earning from primary job"

gen lwage_1y_gm4 = lwage_1y-l4.lwage_1y
label var lwage_1y_gm4 "log annual earning growth from previous m-4 to m"

egen lwage_1y_gm4_sd = sd(lwage_1y_gm4), by(date)
label var lwage_1y_gm4 "std log annual earning growth from previous m-4 to m"

** generate cross-sectional std
egen lwage_1y_sd = sd(lwage_1y), by(date)

** mincer regressions 
reghdfe lwage_1y age age2, a(i.gender i.edu_g) resid
predict lwage_1y_shk, residuals

* including aggregate shock
reghdfe lwage_1y age age2, a(i.date i.gender i.edu_g) resid
predict lwage_1y_id_shk, residuals

gen lwage_1y_ag_shk = lwage_1y_shk- lwage_1y_id_shk

label var lwage_1y_shk "log wage shock"
label var lwage_1y_id_shk "log wage idiosyncratic shock"
label var lwage_1y_ag_shk "log wage aggregate shock"

** first difference
foreach var in lwage_1y_shk lwage_1y_id_shk lwage_1y_ag_shk{
gen `var'_gr = `var'- l4.`var'
}

label var lwage_1y_shk_gr "log growth of unexplained wage"
label var lwage_1y_id_shk_gr "log growth of idiosyncratic unexplained wage"
label var lwage_1y_ag_shk_gr "log growth of aggregate unexplained wage"


** inequality  
gen lwage_1y_shk2 = log(lwage_1y_shk^2)
label var lwage_1y_shk2 "squared log residuals"

** volatility 
gen lwage_1y_shk_gr2 = log(lwage_1y_shk_gr^2)
label var lwage_1y_shk_gr2 "squared log shocks"

egen lwage_1y_shk_gr_sd = sd(lwage_1y_shk_gr), by(date)
label var lwage_1y_shk_gr_sd "standard deviation of log shocks"

gen lwage_1y_shk_gr_var = lwage_1y_shk_gr_sd^2
label var lwage_1y_shk_gr_sd "var of log shocks"


/*
***********************************************
** summary chart of unconditional wages ********
************************************************

preserve

collapse (mean) lwage_1y lwage_1y_sd, by(date year month) 
** average log wage whole sample
twoway  (connected lwage_1y date) if lwage_1y!=., title("The mean of log real wages") 
graph export "${graph_folder}/log_wage_av.png", as(png) replace 

** std log wage whole sample
twoway  (connected lwage_1y_sd date) if lwage_1y_sd!=., title("The standard deviation of log real wages") 
graph export "${graph_folder}/log_wage_sd.png", as(png) replace 
restore 


************************************************
** summary chart of conditional wages ********
************************************************

preserve

collapse (mean) lwage_1y_shk_gr lwage_1y_shk_gr_sd, by(year month date) 
** average log wage shock whole sample
twoway  (connected lwage_1y_shk_gr date) if lwage_1y_shk_gr!=., title("The mean of log real wage shocks") 
graph export "${graph_folder}/log_wage_shk_gr.png", as(png) replace 

** std log wage whole sample
twoway  (connected lwage_1y_shk_gr_sd date) if lwage_1y_shk_gr_sd!=., title("The standard deviation of log real wage shocks") 
graph export "${graph_folder}/log_wage_shk_gr_sd.png", as(png) replace
restore 
*/

***************************************
**** earning level and risk perceptions
***************************************

gen rincsd = sqrt(rincvar)

graph bar rincsd, ///
           over(earning_gp,relabel(1 "10%" 2 "20%" 3 "30%" 4 "40%" 5 "50%" 6 "60%" 7 "70%" 8 "80%" 9 "90%" 10 "100%")) ///
		   bar(1, color(navy)) ///
		   title("Perceived risk by real earning") ///
		   b1title("earning decile") ///
		   ytitle("Average perceived risk (std)") 
graph export "${sum_graph_folder}/boxplot_rvar_earning.png", as(png) replace 


***************************************
**** unemployment experience and risk perceptions
***************************************

/*
graph box rincsd, over(u2e,relabel(1 "other" 2 "recently unemployed")) ///
         title("Perceived risk by unemployment experience") ///
		 ytitle("Average perceived risk (std)") 
graph export "${sum_graph_folder}/boxplot_rvar_ue_peperience.png", as(png) replace 


graph box rincsd, over(u2em12,relabel(1 "other" 2 "unemployed within past year")) ///
         title("Perceived risk by unemployment experience") ///
		 ytitle("Average perceived risk (std)") 
graph export "${sum_graph_folder}/boxplot_rvar_ue_peperience_m12.png", as(png) replace 
*/


************************************************
** income shock /volatility and risk perceptions ********
************************************************

** earning risks 

eststo clear
label var prisk2_all_ch  "change in p risk from m-1 to m"
label var trisk2_all_ch "change in t risk from m-1 to m"
label var rincvar_all_now_ch "change in annual earning risk from m-1 to m"
label var rincvar_sub_now_ch "change in annual group earning risk from m-1 to m"
label var rincvar_all_now "realized annual earning risk"
label var rincvar_sub_now "realized annual group earning risk"

eststo: reghdfe rincvar_rv prisk2_all_ch trisk2_all_ch, a(ID)
eststo: reghdfe rincvar_rv prisk2_sub_ch trisk2_sub_ch, a(ID)

eststo: reghdfe rincvar_rv rincvar_all_now_ch, a(ID)
eststo: reghdfe rincvar_rv rincvar_sub_now_ch, a(ID)

eststo: reghdfe rincvar rincvar_all_now, a(ID)
eststo: reghdfe rincvar rincvar_sub_now, a(ID)

esttab using "${sum_table_folder}/ind/extrapolation_earning_risk.csv", label mtitles se r2 ///
drop(_cons) replace
eststo clear


** separation expectations 

eststo clear

label var u2e "unemployed since m-4"
label var u2em8 "unemployed since m-8"
label var u2em12 "unemployed since y-1"
label var ue_chg "quarterly change in ue"

** seperation risk perceptions and ue experiences 
eststo: reghdfe exp_s_1y_rv exp_s_1y_fe, a(ID)

eststo: reghdfe exp_s_rv exp_s_fe age age2, a(i.gender i.edu_g)  

eststo: reghdfe exp_f_rv exp_f_fe, a(ID)

eststo: reghdfe exp_s_1y_rv i.u2e##c.exp_s_1y_fe, a(ID)

eststo: reghdfe exp_s_1y_rv i.u2em8##c.exp_s_1y_fe, a(ID)

eststo: reghdfe exp_s_1y_rv i.u2em12##c.exp_s_1y_fe, a(ID)

eststo: reghdfe exp_s_1y_rv c.s##c.exp_s_1y_fe, a(ID)

esttab using "${sum_table_folder}/ind/extrapolation_ue_risk.csv", label mtitles se r2 ///
drop(_cons 0.u2e#c.exp_s_1y_fe 0.u2em8#c.exp_s_1y_fe 0.u2em12#c.exp_s_1y_fe) replace
eststo clear


*** earning risks at individual level within SCE  

eststo clear

label var lwage_1y_id_shk_gr "ind shock"
label var lwage_1y_shk_gr "income shock"
label var lwage_1y_shk_gr2 "income shock squared"

** risk perceptions and experienced volatility 

eststo: reghdfe lrincvar lwage_1y_shk_gr2, a(i.date)

eststo: reghdfe lrincvar lwage_1y_shk_gr2, a(i.ID i.date)

** risk perceptions and ue experience 

eststo: reghdfe lrincvar u2e lwage_1y_shk_gr2, a(i.ID)

** risk perceptions and macroeconomy  
eststo: reghdfe lrincvar uerate lwage_1y_shk_gr2, a(i.ID)

esttab using "${sum_table_folder}/ind/extrapolation_earning_risk.csv", label mtitles se r2 ///
drop(_cons age age2) replace
eststo clear

************************************************
** spending decisions and perceived risks **
************************************************
eststo clear
xtset ID date

label var rincvar "perceived earning risk"
label var incvar "perceived earning risk (nominal)"
label var exp_s "perceived ue risk next 4m"
label var exp_s_1y "perceived ue risk next 1y"

eststo: reg spending rincvar

estadd local hast "No",replace
estadd local hasid "No",replace

eststo: areg spending rincvar, a(date)

estadd local hast "Yes",replace
estadd local hasid "No",replace

eststo: areg spending rincvar, a(ID)

estadd local hast "No",replace
estadd local hasid "Yes",replace


eststo: areg spending rincvar i.year, a(ID)

estadd local hast "Yes",replace
estadd local hasid "Yes",replace

eststo: areg spending incvar i.year, a(ID)

estadd local hast "Yes",replace
estadd local hasid "Yes",replace

eststo: areg spending exp_s i.year, a(ID)

estadd local hast "Yes",replace
estadd local hasid "Yes",replace

esttab using "${sum_table_folder}/ind/spending_reg_fe.csv", ///
              label mtitles se  stats(r2 N hast hasid, label("R-squared" "Sample Size" "Time FE" "Individual FE"))  ///
             drop(_cons *.year) replace
eststo clear


************************************************
** pesistence of the individual perceived risks **
************************************************

eststo clear

foreach mom in mean var{
    xtset ID date
	replace Inc_mom = inc`mom' 
    *replace Incmom_ch = Inc`mom'- l1.Inc`mom'

	eststo `mom'1: reg Inc_mom l(1/3).Inc_mom, vce(cluster date)
	eststo `mom'2: reg Inc_mom l(1/6).Inc_mom, vce(cluster date)
	eststo `mom'3: reg Inc_mom l(1/8).Inc_mom, vce(cluster date)
	
	replace Inc_mom = rinc`mom'
	eststo r`mom'1: reg Inc_mom l(1/3).Inc_mom, vce(cluster date)
	eststo r`mom'2: reg Inc_mom l(1/6).Inc_mom, vce(cluster date)
	eststo r`mom'3: reg Inc_mom l(1/8).Inc_mom, vce(cluster date)
 }
 
  
esttab using "${sum_table_folder}/ind/autoregIndM.csv", mtitles se  r2 replace
eststo clear


log close 
