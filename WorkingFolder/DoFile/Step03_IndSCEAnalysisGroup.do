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
xtset ID date   /* this is not correct. ID is unique here.*/
sort ID year month 

*******************************
** Exclude extreme outliers 
******************************

*keep if Q32 < 100 & Q32 >= 10

drop if IncVar < 0


*************************
*** Exclude outliers *****
*************************

local Moments IncMean IncVar IncSkew IncKurt

foreach var in `Moments'{
      egen `var'pl=pctile(`var'),p(5)
	  egen `var'pu=pctile(`var'),p(95)
	  replace `var' = . if `var' <`var'pl | (`var' >`var'pu & `var'!=.)
}


*****************************
*** generate other vars *****
*****************************

gen age_sq = (Q32-30)^2
label var age_sq "Age-squared"

encode _STATE, gen(state_id)
label var state_id "state id"

*****************************
*** generate group vars *****
*****************************

gen cohort = year-Q32
label var cohort "cohort by year of birth"
egen cohort_g = cut(cohort), group(3)
label define cohortlb 0 "1915-1956" 1 "1957-1972" 2 "1973-2000"
label value cohort_g cohortlb

egen age_g = cut(Q32), group(3)  
label var age_g "age group"
label define agelb 0 "Young" 1 "Middle-age" 2 "Old"
label value age_g agelb

egen edu_g = cut(Q36), group(3) 
label var edu_g "education group"
label define edulb 0 "Low Education" 1 "Medium Education" 2 "High Education"
label value edu_g edulb

gen gender_g = Q33 
label var gender_g "gender_grou"
label define gdlb 0 "Male" 1 "Female" 
label value gender_g gdlb

egen inc_g = cut(Q47), group(3)
label var inc_g "income_g"
label define inclb 0 "Low income" 1 "Middle Income" 2 "High Income"
label value inc_g inclb

local group_vars age_g edu_g inc_g cohort_g


**********************************
*** tables and hists of Vars *****
**********************************


foreach gp in `group_vars' {
tabstat Q24_mean Q24_var Q24_iqr IncMean IncVar, st(p10 p50 p90) by(`gp')
}


foreach gp in `group_vars' {
table `gp', c(median Q24_var) by(year)
}


/*

foreach mom in iqr var mean {

twoway (hist Q24_`mom',fcolor(ltblue) lcolor(none)), ///
	   ytitle("") ///
	   title("`mom'")
graph export "${sum_graph_folder}/hist/hist_`mom'.png",as(png) replace  

}

foreach gp in `group_vars' {
foreach mom in iqr var mean {

twoway (hist Q24_`mom' if `gp'==0,fcolor(gs15) lcolor("")) /// 
       (hist Q24_`mom' if `gp'==1,fcolor(ltblue) lcolor("")) ///
	   (hist Q24_`mom' if `gp'==2,fcolor(red) lcolor("")), ///
	   xtitle("") ///
	   ytitle("") ///
	   title("`mom'") ///
	   legend(label(1 `gp'=0) label(2 `gp'=1) label(3 `gp'=2) col(1))

graph export "${sum_graph_folder}/hist/hist_`mom'_`gp'.png",as(png) replace  

}
}


*/

foreach mom in Mean Var Skew Kurt{

twoway (hist Inc`mom',fcolor(ltblue) lcolor(none)), ///
	   ytitle("") ///
	   title("`mom'")
graph export "${sum_graph_folder}/hist/hist_Inc`mom'.png",as(png) replace  

}

foreach gp in `group_vars' {
foreach mom in Mean Var Skew Kurt{

twoway (hist Inc`mom' if `gp'==0,fcolor(gs15) lcolor("")) /// 
       (hist Inc`mom' if `gp'==1,fcolor(ltblue) lcolor("")) ///
	   (hist Inc`mom' if `gp'==2,fcolor(red) lcolor("")), ///
	   xtitle("") ///
	   ytitle("") ///
	   title("`mom'") ///
	   legend(label(1 `gp'=0) label(2 `gp'=1) label(3 `gp'=2) col(1))

graph export "${sum_graph_folder}/hist/hist_Inc_`mom'_`gp'.png",as(png) replace  

}
}

*******************
*** Seasonal ******
*******************


eststo clear

foreach mom in var iqr mean{
eststo: reg Q24_`mom' i.month, robust 
}
foreach mom in Mean Var Skew Kurt{
eststo: reg Inc`mom' i.month, robust 
}

esttab using "${sum_table_folder}/month_fe.csv", ///
             se r2 drop(_cons) ///
			 label replace



********************
** Regression ******
********************

global other_control i.Q33 i.Q34 Q35_1 Q35_2 Q35_3 Q35_4 Q35_5 Q35_6 
global macro_ex_var Q4new Q6new Q9_mean Q13new

eststo clear

foreach mom in var iqr mean{
eststo: reg Q24_`mom' i.age_g i.edu_g i.inc_g i.cohort_g i.year i.state_id, robust 
eststo: reg Q24_`mom' i.age_g i.edu_g i.inc_g i.cohort_g i.year i.state_id ${other_control}, robust 
eststo: reg Q24_`mom' i.age_g i.edu_g i.inc_g i.cohort_g i.year i.state_id ${other_control} ${macro_ex_var}, robust 
}
foreach mom in Mean Var Skew Kurt{
eststo: reg Inc`mom' i.age_g i.edu_g i.inc_g i.cohort_g i.year i.state_id, robust 
eststo: reg Inc`mom' i.age_g i.edu_g i.inc_g i.cohort_g i.year i.state_id ${other_control}, robust 
eststo: reg Inc`mom' i.age_g i.edu_g i.inc_g i.cohort_g i.year i.state_id ${other_control} ${macro_ex_var}, robust 

}

esttab using "${sum_table_folder}/mom_group.csv", ///
             se r2 drop(0.age_g 0.edu_g 0.inc_g 0.cohort_g  *.year *state_id 1.Q33 1.Q34 _cons) ///
			 label replace

			 
			 
log close 
