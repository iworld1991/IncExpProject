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

************************************
**  Collapse to Population Series **
************************************

collapse (median)  IncMean IncVar Q24_mean Q24_var Q24_iqr, by(year month date) 
order date year month
duplicates report date 

drop date 
gen date_str=string(year)+"m"+string(month)
gen date= monthly(date_str,"YM")
format date %tm
tsset date 

************************************
**  Times Series Plots  **
************************************

foreach mom in mean var iqr{
tsline Q24_`mom',lwidth(thick) title("`mom'") ///
       ytitle("`mom'") 
	   
graph export "${sum_graph_folder}/median_`mom'.png",as(png) replace  
}

log close 
