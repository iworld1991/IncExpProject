clear
global mainfolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/Graphs/pop"
global sum_table_folder "${mainfolder}/Tables"

cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/popSCE_log",replace


***************************
**  Clean and Merge Data **
***************************

use "${folder}/SCE/IncExpSCEDstIndM",clear 

duplicates report year month userid


******************************
*** Merge with demographics **
*****************************

merge 1:1 year month userid using "${folder}/SCE/IncExpSCEProbIndM",keep(master match) 
rename _merge hh_info_merge


******************************
*** drop states with two few obs
*******************************

bysort statecode date: gen obs_ct = _N
*drop if obs_ct <=5

************************************
**  Collapse to Population Series **
************************************

collapse (median)  Q24_mean Q24_var Q24_iqr IncMean IncVar IncSkew IncKurt wagegrowth unemp_rate, by(state statecode year month date) 
order state date year month
duplicates report date 

drop date 
gen date_str=string(year)+"m"+string(month)
gen date= monthly(date_str,"YM")
format date %tm
xtset statecode date 


************************************
**  generate lag variables **
************************************

gen wagegrowthl1 = l2.wagegrowth
label var wagegrowthl "recent wage growth in the state"

gen unemp_rate_ch = unemp_rate -l1.unemp_rate
label var unemp_rate_ch "chagne in uemp rate"

/*
************************************
**  scatter plots  **
************************************

foreach mom in var iqr{
 twoway (scatter unemp_rate Q24_`mom' ) ///
        (lfit unemp_rate Q24_`mom',lcolor(red)), ///
		title("Average perceived risk and regional labor market condition") ///
		xtitle("state uemp rate") ///
		ytitle("perceived risk")
graph export "${sum_graph_folder}/scatter_`mom'_unemp_rate.png",as(png) replace  
}


foreach mom in var iqr{
 twoway (scatter wagegrowthl Q24_`mom' ) ///
        (lfit wagegrowthl Q24_`mom',lcolor(red)), ///
		title("Average perceived risk and regional labor market condition") ///
		xtitle("state wage growth") ///
		ytitle("perceived risk")
graph export "${sum_graph_folder}/scatter_`mom'_wagegrowth.png",as(png) replace  
}

*/

************************************
**  regression results       **
************************************

eststo clear

foreach mom in var iqr{
eststo: xtreg Q24_`mom' l(0/1)wagegrowth i.date,fe
eststo: xtreg Q24_`mom' l(0/1)unemp_rate_ch i.date,fe
}

esttab using "${sum_table_folder}/mom_group_state.csv", ///
             se r2 drop(_cons *.date) ///
			 label replace

log close 
