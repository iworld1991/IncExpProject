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


local Moments varMean iqrMean rvarMean skewMean varMed iqrMed rvarMed skewMed

foreach mom in `Moments'{
gen `mom'_ch = `mom' - l1.`mom'
label var `mom'_ch "change in `mom'"
}



************************
***** regression *******
***********************


foreach mom in `Moments'{
forvalues i=1(1)12{
newey  `mom'  f`i'.sp500, lag(18)
return list 
local id = `i'+1
putexcel set "${sum_table_folder}/macro_corr_stata.xls", sheet("`mom'") modify 
local coeff = _b[f`i'.sp500]
local coeff : display %5.3f `coeff'
local se = _se[f`i'.sp500]
local t = `coeff'/`se'
local p = 2*ttail(e(df_r),abs(`t'))

local star = ""
if `p'< =0.01 local star = "***"
if `p'<=0.05 & `p'> 0.01 local star = "**"
if `p' <=0.1 & `p'> 0.05 local star = "*"
*local coeff_str: display `coeff' `star'

putexcel A`id' =(`i')
putexcel B`id' = (`coeff')
*putexcel C`id' = (`se')
*putexcel D`id' =(`p')
putexcel C`id' =("`star'")

}
putexcel A1 =("# months ahead")
putexcel B1 = ("coefficients")
putexcel C1 = ("star")
putexcel D1 = ("`mom'")

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

log close 
