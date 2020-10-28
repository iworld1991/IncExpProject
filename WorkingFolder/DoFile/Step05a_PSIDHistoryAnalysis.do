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

import excel "${other}PSID/psid_history_vol3.xls", sheet("Sheet1") firstrow
destring year cohort rmse, force replace

gen rmseqrt = sqrt(rmse)
*hist rmseqrt

***********************
** generate variables 
**************************

gen age = year-cohort + 21
label var age "age"

***********************
** relabel ************
**********************

label var N "history sample size"
label var R2 "history r-square"
label var rmse "experienced volatility"
label var rmseqrt "experienced volatility std."


***********************
** relabel ************
**********************


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
merge 1:m Q32 year using "${folder}/SCE/IncExpSCEProbIndM", keep(using match) 
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

keep if age > 20 & age < 65
  
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

foreach var in Q24_var Q24_iqr rmse{
gen l`var' = log(`var')
}

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

preserve
bysort year age: gen ct = _N

collapse lQ24_var lrmse, by(year age) 

label var lQ24_var "Perceived risk"
label var lrmse "Experienced volatility"
twoway (scatter lQ24_var lrmse, color(ltblue)) ///
       (lfit lQ24_var lrmse, lcolor(red) lw(thick) lpattern(dash)) if lrmse!=., ///
	   title("Experienced volatility and perceived income risks") ///
	   xtitle("log experienced volatility") ///
	   ytitle("log perceived income risks") ///
	   legend(off)
graph export "${sum_graph_folder}/experience_var_var_data.png", as(png) replace 
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

*/

*********************************************
** experienced volatility and perceived risk regression
*******************************************
label var lQ24_var "log perceived risk"
label var lQ24_iqr "log perceived iqr"

eststo clear
foreach var in lQ24_var lQ24_iqr{
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

