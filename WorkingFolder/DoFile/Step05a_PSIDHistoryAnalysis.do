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
** experienced volatility and perceived risk
*******************************************

*****************
** chart 
*****************



*****************
** regression 
*****************



save "${mainfolder}/OtherData/psid_history_vols.dta", replace 


