clear
global mainfolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/Graphs/ind"
global sum_table_folder "${mainfolder}/Tables"

cd ${folder}
pwd
set more off 
capture log close


log using "${mainfolder}/indSCE_Maro_log",replace

use "${folder}/SCE/IncExpSCEIndMacroM.dta"


************************
***** date *************
***********************
generate new_date = dofc(date)
gen year = year(new_date)
gen month = month(new_date)
gen date_str = string(year)+ "m"+string(month)
drop date index new_date 
gen date = monthly(date_str,"YM")
format date %tm
drop date_str
order userid date 
xtset userid date 


************************
***** regression *******
***********************

local Moments incexp rincexp incvar rincvar inciqr incskew 


foreach mom in `Moments'{
*reg `mom'  c.l1.sp500##i.byear_gr c.l1.sp500##i.HHinc_gr c.l1.sp500##i.educ_gr c.l1.sp500##i.age_gr
reg `mom'  c.Stkprob##i.byear_gr c.Stkprob##i.HHinc_gr c.Stkprob##i.educ_gr c.Stkprob##i.age_gr

}


log close 
