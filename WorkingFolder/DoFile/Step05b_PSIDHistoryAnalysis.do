clear
global mainfolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder"
global folder "${mainfolder}/SurveyData/"
global other "${mainfolder}/OtherData/"
global sum_graph_folder "${mainfolder}/Graphs/ind"
global sum_table_folder "${mainfolder}/Tables/"

cd ${folder}
pwd
set more off 
capture log close

import excel "${other}PSID/psid_history_vol_test_decomposed_edu.xlsx", sheet("Sheet1") firstrow

destring year cohort av_gr var_shk av_id_gr var_id_shk av_ag_gr var_ag_shk permanent transitory N, force replace

***********************
** generate variables 
************************

gen age = year-cohort + 20
label var age "age"

***********************
** relabel ************
**********************

label var N "history sample size"
label var av_gr "experienced log unexplained income growth"
label var var_shk "experienced volatility"
label var av_id_gr "experienced log unexplained income growth (idiosyncratic)"
label var var_id_shk "experienced volatility (idiosyncratic)"
label var av_ag_gr "experienced log unexplained income growth (aggregate)"
label var var_ag_shk "experienced volatility (aggregate)"
label var permanent "experienced permanent volatility std"
label var transitory "experienced transitory volatility std"

***********************************
** extend psid history data to 2019 
************************************

/*
expand 3 if year==2017
sort cohort year 
replace year = 2018 if year==2017 & year[_n-1]==2017 & cohort ==cohort[_n-1]
replace year = 2019 if year==2017 & year[_n-1]==2018 & cohort ==cohort[_n-1]
*/

***********************************
** merge with perceived risk data 
************************************

gen Q32 = age
gen educ_gr = edu

merge 1:m Q32 year educ_gr using "${folder}/SCE/IncExpSCEProbIndM", keep(using match) 
rename _merge sce_ind_merge 

merge 1:1 year month userid using "${folder}/SCE/IncExpSCEDstIndM", keep(using match) 
rename _merge sce_ind_merge2

***********************
** format the date 
**********************

drop date 
gen date_str=string(year)+"m"+string(month) 
gen date= monthly(date_str,"YM")
format date %tm
order userid date year month   



***********************
** filters
**********************

keep if age > 20 & age <= 55
 
*********************************************
** generate new group variables 
*******************************************

** income group 
egen inc_gp = cut(Q47), group(3) 

** finanial condition improvement 
gen Q1_gp = .
replace Q1_gp =1 if Q1<=2
replace Q1_gp =2 if Q1==3
replace Q1_gp =3 if Q1>3 & Q1!=.

** cohort group

egen cohort_gp = cut(cohort), at(1970,1980,1990,2000,2010,2020)
label var cohort_gp "cohort"

** age group
egen age_gp = cut(age), at(20 35 55,70)
label var age_gp "age group"

*********************************************
** generate variables 
*******************************************

foreach var in Q24_var Q24_iqr var_shk var_id_shk var_ag_shk permanent transitory{
gen l`var' = log(`var')
}

gen lprobUE= log(Q4new)
label var lprobUE "log probability of UE higher next year"

*****************
** chart 
*****************

/*
graph box rmse if year == 2017, ///
           over(cohort_gp,relabel(1 "1970" 2 "1980" 3 "1990" 4 "2000" 5 "2010")) ///
		   medline(lcolor(black) lw(thick)) ///
		   box(1,bfcolor(red) blcolor(black)) ///
		   title("Experienced volatility of different cohorts up to 2017") ///
		   b1title("year of entering job market")

graph export "${sum_graph_folder}/experience_var_bycohort.png", as(png) replace 
*/


** different experience of different cohort 

*** by cohort and time

/*
preserve
bysort year age: gen ct = _N

drop if ct<=30

collapse lQ24_var av_gr av_id_gr av_ag_gr lvar_shk lvar_id_shk lvar_ag_shk lpermanent ltransitory ue_av ue_var, by(year age) 

gen pt_ratio = lpermanent-ltransitory
label var pt_ratio "permanent/transitory risk ratio"

label var lQ24_var "Perceived risk"
label var av_gr "Experienced log income change"
label var lvar_shk "Experienced volatility"
label var av_id_gr "Experienced log idiosyncratic change"
label var lvar_id_shk "Experienced idiosyncratic volatility"
label var av_ag_gr "Experienced log aggregate change"
label var lvar_ag_shk "Experienced aggregate volatility"

label var lpermanent "Experienced permanent volatility"
label var ltransitory "Experienced transitory volatility"

label var ue_av "average UE rate"
label var ue_var "volatility of UE rate"

* ag ue
twoway (scatter lQ24_var ue_av, color(ltblue)) ///
       (lfit lQ24_var ue_av, lcolor(red) lw(thick) lpattern(dash)) if ue_av!=., ///
	   title("Experienced UE and perceived income risks") ///
	   xtitle("experienced UE rate") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ue_var_data.png", as(png) replace 

* ag ue var
twoway (scatter lQ24_var ue_var, color(ltblue)) ///
       (lfit lQ24_var ue_var, lcolor(red) lw(thick) lpattern(dash)) if ue_var!=., ///
	   title("Experienced UE volatility and perceived income risks") ///
	   xtitle("experienced UE rate") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ue_var_var_data.png", as(png) replace 


* growth 
twoway (scatter lQ24_var av_gr, color(ltblue)) ///
       (lfit lQ24_var av_gr, lcolor(red) lw(thick) lpattern(dash)) if av_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced income growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_gr_var_data.png", as(png) replace 

* risk 
twoway (scatter lQ24_var lvar_shk, color(ltblue)) ///
       (lfit lQ24_var lvar_shk, lcolor(red) lw(thick) lpattern(dash)) if lvar_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_var_data.png", as(png) replace 

* id growth
twoway (scatter lQ24_var av_id_gr, color(ltblue)) ///
       (lfit lQ24_var av_id_gr, lcolor(red) lw(thick) lpattern(dash)) if av_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced idiosyncratic growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_id_gr_var_data.png", as(png) replace 

* id risk 
twoway (scatter lQ24_var lvar_id_shk, color(ltblue)) ///
       (lfit lQ24_var lvar_id_shk, lcolor(red) lw(thick) lpattern(dash)) if lvar_id_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced idiosyncratic volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_id_var_data.png", as(png) replace 


* ag growth
twoway (scatter lQ24_var av_ag_gr, color(ltblue)) ///
       (lfit lQ24_var av_ag_gr, lcolor(red) lw(thick) lpattern(dash)) if av_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced aggregate growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ag_gr_var_data.png", as(png) replace 

* ag risk 
twoway (scatter lQ24_var lvar_ag_shk, color(ltblue)) ///
       (lfit lQ24_var lvar_ag_shk, lcolor(red) lw(thick) lpattern(dash)) if lvar_ag_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced aggregate volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_ag_var_data.png", as(png) replace 

* permanent risk
twoway (scatter lQ24_var lpermanent, color(ltblue)) ///
       (lfit lQ24_var lpermanent, lcolor(red) lw(thick) lpattern(dash)) if lpermanent!=., ///
	   title("Experienced permanent volatility and perceived income risks") ///
	   xtitle("log experienced permanent volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_permanent_var_data.png", as(png) replace 

* transitory risk
twoway (scatter lQ24_var ltransitory, color(ltblue)) ///
       (lfit lQ24_var ltransitory, lcolor(red) lw(thick) lpattern(dash)) if ltransitory!=., ///
	   title("Experienced transitory volatility and perceived income risks") ///
	   xtitle("log experienced transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_transitory_var_data.png", as(png) replace 

* permanent/transitory ratio

twoway (scatter lQ24_var pt_ratio, color(ltblue)) ///
       (lfit lQ24_var pt_ratio, lcolor(red) lw(thick) lpattern(dash)) if pt_ratio!=., ///
	   title("Experienced volatility ratio and perceived income risks") ///
	   xtitle("log experienced permanent/transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_ratio_var_data.png", as(png) replace 

restore
*/

*** by cohort/time/educ

preserve
bysort year age educ_gr: gen ct = _N

drop if ct<=30

collapse lQ24_var av_gr av_id_gr av_ag_gr lvar_shk lvar_id_shk lvar_ag_shk lpermanent ltransitory ue_av ue_var, by(year age educ_gr) 

gen pt_ratio = lpermanent-ltransitory
label var pt_ratio "permanent/transitory risk ratio"

label var lQ24_var "Perceived risk"
label var av_gr "Experienced log income change"
label var lvar_shk "Experienced volatility"
label var av_id_gr "Experienced log idiosyncratic change"
label var lvar_id_shk "Experienced idiosyncratic volatility"
label var av_ag_gr "Experienced log aggregate change"
label var lvar_ag_shk "Experienced aggregate volatility"

label var lpermanent "Experienced permanent volatility"
label var ltransitory "Experienced transitory volatility"

label var ue_av "average UE rate"
label var ue_var "volatility of UE rate"


* ag ue
twoway (scatter lQ24_var ue_av, color(ltblue)) ///
       (lfit lQ24_var ue_av, lcolor(red) lw(thick) lpattern(dash)) if ue_av!=., ///
	   title("Experienced UE and perceived income risks") ///
	   xtitle("experienced UE rate") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ue_var_data.png", as(png) replace 

* ag ue var
twoway (scatter lQ24_var ue_var, color(ltblue)) ///
       (lfit lQ24_var ue_var, lcolor(red) lw(thick) lpattern(dash)) if ue_var!=., ///
	   title("Experienced UE volatility and perceived income risks") ///
	   xtitle("experienced UE rate") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ue_var_var_data.png", as(png) replace 

* growth 
twoway (scatter lQ24_var av_gr, color(ltblue)) ///
       (lfit lQ24_var av_gr, lcolor(red) lw(thick) lpattern(dash)) if av_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced income growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_gr_var_data.png", as(png) replace 

* risk 
twoway (scatter lQ24_var lvar_shk, color(ltblue)) ///
       (lfit lQ24_var lvar_shk, lcolor(red) lw(thick) lpattern(dash)) if lvar_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_var_data.png", as(png) replace 

* id growth
twoway (scatter lQ24_var av_id_gr, color(ltblue)) ///
       (lfit lQ24_var av_id_gr, lcolor(red) lw(thick) lpattern(dash)) if av_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced idiosyncratic growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_id_gr_var_data.png", as(png) replace 

* id risk 
twoway (scatter lQ24_var lvar_id_shk, color(ltblue)) ///
       (lfit lQ24_var lvar_id_shk, lcolor(red) lw(thick) lpattern(dash)) if lvar_id_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced idiosyncratic volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_id_var_data.png", as(png) replace 


* ag growth
twoway (scatter lQ24_var av_ag_gr, color(ltblue)) ///
       (lfit lQ24_var av_ag_gr, lcolor(red) lw(thick) lpattern(dash)) if av_gr!=., ///
	   title("Experienced income growth and perceived income risks") ///
	   xtitle("experienced aggregate growth") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_ag_gr_var_data.png", as(png) replace 

* ag risk 
twoway (scatter lQ24_var lvar_ag_shk, color(ltblue)) ///
       (lfit lQ24_var lvar_ag_shk, lcolor(red) lw(thick) lpattern(dash)) if lvar_ag_shk!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced aggregate volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_ag_var_data.png", as(png) replace 

* permanent risk
twoway (scatter lQ24_var lpermanent, color(ltblue)) ///
       (lfit lQ24_var lpermanent, lcolor(red) lw(thick) lpattern(dash)) if lpermanent!=., ///
	   title("Experienced permanent volatility and perceived income risks") ///
	   xtitle("log experienced permanent volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_permanent_var_data.png", as(png) replace 

* transitory risk
twoway (scatter lQ24_var ltransitory, color(ltblue)) ///
       (lfit lQ24_var ltransitory, lcolor(red) lw(thick) lpattern(dash)) if ltransitory!=., ///
	   title("Experienced transitory volatility and perceived income risks") ///
	   xtitle("log experienced transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_transitory_var_data.png", as(png) replace 

* permanent/transitory ratio

twoway (scatter lQ24_var pt_ratio, color(ltblue)) ///
       (lfit lQ24_var pt_ratio, lcolor(red) lw(thick) lpattern(dash)) if pt_ratio!=., ///
	   title("Experienced volatility ratio and perceived income risks") ///
	   xtitle("log experienced permanent/transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_ratio_var_data.png", as(png) replace 
restore

*** by age only 
preserve
bysort age: gen ct = _N

collapse lQ24_var av_gr av_id_gr av_ag_gr lvar_shk lvar_id_shk lvar_ag_shk lpermanent ltransitory, by(year age) 

gen pt_ratio = lpermanent-ltransitory
label var pt_ratio "permanent/transitory risk ratio"

label var lQ24_var "Perceived risk"
label var av_gr "Experienced log income change"
label var lvar_shk "Experienced volatility"
label var av_id_gr "Experienced log idiosyncratic change"
label var lvar_id_shk "Experienced idiosyncratic volatility"
label var av_ag_gr "Experienced log aggregate change"
label var lvar_ag_shk "Experienced aggregate volatility"

label var lpermanent "Experienced permanent volatility"
label var ltransitory "Experienced transitory volatility"

twoway (scatter lQ24_var lpermanent, color(ltblue)) ///
       (lfit lQ24_var lpermanent, lcolor(red) lw(thick) lpattern(dash)) if lpermanent!=., ///
	   title("Experienced permanent volatility and perceived income risks") ///
	   xtitle("log experienced permanent volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_permanent_var_data_by_age.png", as(png) replace 

twoway (scatter lQ24_var ltransitory, color(ltblue)) ///
       (lfit lQ24_var ltransitory, lcolor(red) lw(thick) lpattern(dash)) if ltransitory!=., ///
	   title("Experienced transitory volatility and perceived income risks") ///
	   xtitle("log experienced transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_transitory_var_data_by_age.png", as(png) replace 

twoway (scatter lQ24_var pt_ratio, color(ltblue)) ///
       (lfit lQ24_var pt_ratio, lcolor(red) lw(thick) lpattern(dash)) if pt_ratio!=., ///
	   title("Experienced volatility ratio and perceived income risks") ///
	   xtitle("log experienced permanent/transitory volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_ratio_var_data_by_age.png", as(png) replace 

restore 

/*
preserve

bysort year age: gen ct = _N
collapse lQ24_var lrmse, by(cohort inc_gp) 

label var lQ24_var "log perceived risk"
label var lrmse "log experienced volatility"
twoway (scatter lQ24_var lrmse , color(ltblue)) ///
       (lfit lQ24_var lrmse, lcolor(red) lw(thick)) if lrmse!=., ///
	   by(inc_gp,title("Experienced volatility and perceived income risks") note("Graph by income group") rows(1)) ///
	   xtitle("log experienced volatility") ///
	   ytitle("log perceived income riks") ///
	   legend(off)
	   
graph export "${sum_graph_folder}/experience_var_var_by_income_data.png", as(png) replace 
restore


*********************************************
** experienced volatility and perceived risk regression
*******************************************

label var lQ24_var "log perceived risk"
label var lQ24_iqr "log perceived iqr"
label var lprobUE "log prob of higher UE"

eststo clear
foreach var in lQ24_var lQ24_iqr lprobUE{
eststo: reg `var' lrmse i.age_gp
estadd local hasage "Yes",replace
estadd local haseduc "No",replace
estadd local hasinc "No",replace


eststo: reg `var' lrmse i.age_gp i.Q36
estadd local hasage "Yes",replace
estadd local haseduc "Yes",replace
estadd local hasinc "No",replace

eststo: reg `var' lrmse i.age_gp i.Q36 i.inc_gp
estadd local hasage "Yes",replace
estadd local haseduc "Yes",replace
estadd local hasinc "Yes",replace

}

label var lrmse "log experienced volatility"
esttab using "${sum_table_folder}/micro_reg_history_vol.csv", ///
         keep(lrmse) st(r2 N hasage haseduc hasinc,label("R-squre" "N" "Control age" "Control educ" "Control income")) ///
		 label ///
		 replace 
		 
		 
************************************************
**  experienced volatility and state wage growth
***********************************************

label var lQ24_var "log perceived risk"
label var lQ24_iqr "log perceived iqr"

eststo clear
foreach var in lQ24_var lQ24_iqr{

eststo: reg `var' c.lrmse##c.wagegrowth i.age_gp i.Q36 i.inc_gp
estadd local hasage "Yes",replace
estadd local haseduc "Yes",replace
estadd local hasinc "Yes",replace

}

label var lrmse "log experienced volatility"
esttab using "${sum_table_folder}/micro_reg_history_vol_state.csv", ///
         keep(lrmse *lrmse) st(r2 N hasage haseduc hasinc,label("R-squre" "N" "Control age" "Control educ" "Control income")) ///
		 label ///
		 replace 

	*/
	
************************************************
**  experienced volatility and numeracy
***********************************************


label var lQ24_var "log perceived risk"
label var lQ24_iqr "log perceived iqr"

eststo clear
foreach var in lQ24_var lQ24_iqr{

eststo: reg `var' c.lrmse##c.nlit i.age_gp i.Q36 i.inc_gp
estadd local hasage "Yes",replace
estadd local haseduc "Yes",replace
estadd local hasinc "Yes",replace

}

label var lrmse "log experienced volatility"
esttab using "${sum_table_folder}/micro_reg_history_vol_nlit.csv", ///
         keep(lrmse *lrmse) st(r2 N hasage haseduc hasinc,label("R-squre" "N" "Control age" "Control educ" "Control income")) ///
		 label ///
		 replace 
