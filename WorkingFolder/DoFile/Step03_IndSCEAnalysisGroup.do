clear
global mainfolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/Graphs/ind"
global sum_table_folder "${mainfolder}/Tables"

cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/indSCE_Est_log",replace


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

*******************************
**  Set Panel Data Structure **
*******************************
rename userid ID 
xtset ID date
sort ID year month 


*****************************
*** generate group vars *****
*****************************

egen age_g = cut(Q32), group(3)  
label var age_g "age group"

egen edu_g = cut(Q36), group(3) 
label var edu_g "education group"

gen gender_g = Q33 
label var gender_g "gender_grou"

egen inc_g = cut(Q47), group(3)
label var inc_g "income_g"


local group_vars age_g edu_g inc_g


**********************************
*** tables and hists of Vars *****
**********************************


foreach gp in `group_vars' {
tabstat Q24_mean Q24_var Q24_iqr, st(p10 p50 p90) by(`gp')
}


foreach gp in `group_vars' {
foreach mom in iqr var mean {

twoway (hist Q24_`mom' if `gp'==0,fcolor(gs15) lcolor(none)) /// 
       (hist Q24_`mom' if `gp'==1,fcolor(ltblue) lcolor(none)) ///
	   (hist Q24_`mom' if `gp'==2,fcolor(red) lcolor(none)), ///
	   xlabel("") ///
	   ylabel("") ///
	   title("`mom'") ///
	   legend(label(1 `gp'=0) label(2 `gp'=1) label(3 `gp'=2) col(1))

graph export "${sum_graph_folder}/hist/hist_`mom'_`gp'.png",as(png) replace  

}
}

log close 
