clear
global mainfolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/Graphs/ind"
global sum_table_folder "${mainfolder}/Tables/"

cd ${folder}
pwd
set more off 
capture log close

import excel "${sum_table_folder}psid/psid_history_vol.xls", sheet("Sheet1") firstrow
destring year cohort rmse, force replace

gen rmseqrt = sqrt(rmse)
*hist rmseqrt

***********************
** generate variables 
**************************

gen age = year-cohort + 22 
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

*********************************************
** experienced volatility and perceived risk
*******************************************
label var Q24_var "Perceived risk"
label var Q24_var "Perceived iqr"

eststo clear
foreach var in Q24_var Q24_iqr{
eststo: reg `var' rmse i.age i.Q36 i.inc_gp
estadd local hasT "No",replace
eststo: reg `var' rmse i.age i.Q36 i.inc_gp i.date  
estadd local hasT "Yes",replace
}
esttab, keep(rmse) st(r2 N hasT,label("R-squre" "N" "TimeFE")) label 


*****************
** chart 
*****************


*****************
** regression 
*****************



save "${mainfolder}/OtherData/SCEM_PSID.dta", replace 


