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

use "${folder}/SCE/IncExpSCEPopMacroM.dta"


************************
***** date *************
***********************
generate new_date = dofc(index)
gen year = year(new_date)
gen month = month(new_date)
gen date_str = string(year)+ "m"+string(month)
drop index new_date 
gen date = monthly(date_str,"YM")
format date %tm
drop date_str
tsset date 

**********************
***** Other measures **
************************


local Moments varMean iqrMean rvarMean skewMean

foreach mom in `Moments'{
gen `mom'_ch = `mom' - l1.`mom'
label var `mom'_ch "change in `mom'"
}


************************
***** correlation plot *******
***********************

foreach mom in `Moments'{
xcorr `mom' sp500, lag(12) title("`mom' and stock market")
*reg `mom'  c.l1.sp500##i.byear_gr c.l1.sp500##i.HHinc_gr c.l1.sp500##i.educ_gr c.l1.sp500##i.age_gr
*reg `mom'  c.Stkprob##i.byear_gr c.Stkprob##i.HHinc_gr c.Stkprob##i.educ_gr c.Stkprob##i.age_gr
graph export "${sum_graph_folder}/ts/corr_`mom'_stk.png",as(png) replace 
}


************************
***** regression *******
***********************


foreach mom in `Moments'{
newey  `mom'  f12.sp500, lag(12)
*reg `mom'  c.l1.sp500##i.byear_gr c.l1.sp500##i.HHinc_gr c.l1.sp500##i.educ_gr c.l1.sp500##i.age_gr
*reg `mom'  c.Stkprob##i.byear_gr c.Stkprob##i.HHinc_gr c.Stkprob##i.educ_gr c.Stkprob##i.age_gr

}


ddd

log close 
