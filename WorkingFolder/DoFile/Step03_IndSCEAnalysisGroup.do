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

*********************************************************
*** before working with SCE, clean the stock market data 
********************************************************

use "${mainfolder}/OtherData/sp500.dta",clear 

generate new_date = dofc(DATE)
*format new_date %tm
gen year = year(new_date)
gen month = month(new_date)
gen date_str = string(year)+ "m"+string(month)
gen date = monthly(date_str,"YM")
format date %tm
drop DATE date_str year month new_date 
label var sp500 "growth rate (%) of sp500 index from last month"
save "${mainfolder}/OtherData/sp500M.dta",replace 
clear 


***************************
**  Clean and Merge Data **
***************************

use "${folder}/SCE/IncExpSCEDstIndM",clear 

duplicates report year month userid

************************************************
*** Merge with demographics and other moments **
************************************************

merge 1:1 year month userid using "${folder}/SCE/IncExpSCEProbIndM",keep(master match) 
rename _merge hh_info_merge

** format the date 
drop date 
gen date_str=string(year)+"m"+string(month) 
gen date= monthly(date_str,"YM")
format date %tm
order userid date year month   

*************************************
*** Merge with stock market data   **
*************************************

merge m:1 date using "${mainfolder}/OtherData/sp500M.dta", keep(master match) 
rename _merge sp_merge

*******************************
**  Set Panel Data Structure **
*******************************
rename userid ID 
xtset ID date   /* this is not correct. ID is unique here.*/
sort ID year month 

*******************************
** Exclude extreme outliers 
******************************

keep if Q32 <= 65 & Q32 >= 20

*****************************************
****  Renaming so that more consistent **
*****************************************

rename Q24_mean incmean
rename Q24_var incvar
rename Q24_iqr inciqr
rename IncSkew incskew 
rename Q24_rmean rincmean
rename Q24_rvar rincvar

rename Q36 educ
rename D6 HHinc 
rename Q32 age 
rename Q33 gender 
rename Q10_1 fulltime
rename Q10_2 parttime
rename Q12new selfemp
rename Q6new Stkprob
rename Q4new UEprobAgg
rename Q13new UEprobInd
rename Q26v2 spending_dum
rename Q26v2part2 spending 


************************
** focus on non-zero skewness
****************************

replace incskew = . if incskew==0


*************************
*** Exclude outliers *****
*************************

local Moments incmean rincmean incvar rincvar inciqr incskew 

foreach var in `Moments'{
      egen `var'pl=pctile(`var'),p(1)
	  egen `var'pu=pctile(`var'),p(99)
	  replace `var' = . if `var' <`var'pl | (`var' >`var'pu & `var'!=.)
}

/*
* other thresholds 

foreach var in `Moments'{
      egen `var'l_truc=pctile(`var'),p(8)
	  egen `var'u_truc=pctile(`var'),p(92)
	  replace `var' = . if `var' <`var'l_truc | (`var' >`var'u_truc & `var'!=.)
}
*/

*****************************
*** generate other vars *****
*****************************

gen age_sq = age^2
label var age_sq "Age-squared"

encode state, gen(state_id)
label var state_id "state id"

*****************************
*** generate group vars *****
*****************************

*egen byear_g = cut(byear), group(4)

gen byear_g = cond(byear>=1980,1,0)
label define byearglb 0 "before 1980s" 1 "after 1980s"
*label define byearglb 0 "1950s" 1 "1960s" 2 "1970s" 3 "1980s"
label value byear_g byearlb

egen age_g = cut(age), group(3)  
label var age_g "age group"
label define agelb 0 "young" 1 "middle-age" 2 "old"
label value age_g agelb

egen edu_g = cut(educ), group(2) 
label var edu_g "education group"
label define edulb 0 "low educ" 1 "high educ" 
label value edu_g edulb

label define gdlb 0 "Male" 1 "Female" 
label value gender gdlb

egen HHinc_g = cut(HHinc), group(2)
label var HHinc_g "Household income group"
label define HHinc_glb 0 "low inc" 1 "high inc"
label value HHinc_g HHinc_glb

label define gender_glb 1 "male" 2 "female"
label value gender gender_glb

gen fbetter =cond(Q1>2,1,0)
replace fbetter = . if Q1 ==3  
label var fbetter "finance better"

label define better_glb 0 "worse" 1 "better"
label value fbetter better_glb

gen nlit_g = cond(nlit>=3,1,0) 
replace nlit_g = . if nlit ==.
label var nlit_g "numeracy literacy score group"
label define nlilb 0 "low" 1 "high" 
label value nlit_g nlitlb

local group_vars byear_g age_g edu_g HHinc_g fbetter nlit_g



*********************************
*** bar charts *****
**********************************


graph bar incvar, ///
           over(HHinc,relabel(1 "<10k" 2 "<20k" 3 "<30k" 4 "<40k" 5 "<50k" 6 "<60k" 7 "<75k" 8 "<100k" 9 "<150k" 10 "<200k" 11 ">200k")) ///
		   bar(1, color(navy)) ///
		   title("Perceived Risk by household income") ///
		   b1title("Household income") ///
		   ytitle("Average perceived risk") 
graph export "${sum_graph_folder}/boxplot_var_HHinc_stata.png", as(png) replace 


**********************************
*** tables and hists of Vars *****
**********************************

/*
local Moments incmean incvar inciqr rincmean rincvar incskew

foreach gp in `group_vars' {
tabstat `Moments', st(p10 p50 p90) by(`gp')
}


foreach gp in `group_vars' {
table `gp', c(median incvar mean incvar median rincvar mean rincvar) by(year)
}


** histograms 

foreach mom in `Moments'{

twoway (hist `mom',fcolor(ltblue) lcolor(none)), ///
	   ytitle("") ///
	   title("`mom'")
graph export "${sum_graph_folder}/hist/hist_`mom'.png",as(png) replace  

}


* 4 groups 
foreach gp in byear_g{
foreach mom in `Moments'{
twoway (hist `mom' if `gp'==0,fcolor(gs15) lcolor("")) /// 
       (hist `mom' if `gp'==1,fcolor(ltblue) lcolor("")) ///
	   (hist `mom' if `gp'==2,fcolor(red) lcolor("")) ///
	   (hist `mom' if `gp'==3,fcolor(green) lcolor("")), ///
	   xtitle("") ///
	   ytitle("") ///
	   title("`mom'") ///
	   legend(label(1 `gp'=0) label(2 `gp'=1) label(3 `gp'=2) label(4 `gp'=3) col(1))

graph export "${sum_graph_folder}/hist/hist_`mom'_`gp'.png",as(png) replace  
}
}

* 3 groups 
foreach gp in HHinc_g age_g{
foreach mom in `Moments'{

twoway (hist `mom' if `gp'==0,fcolor(gs15) lcolor("")) /// 
       (hist `mom' if `gp'==1,fcolor(ltblue) lcolor("")) ///
	   (hist `mom' if `gp'==2,fcolor(red) lcolor("")), ///
	   xtitle("") ///
	   ytitle("") ///
	   title("`mom'") ///
	   legend(label(1 `gp'=0) label(2 `gp'=1) label(3 `gp'=2) col(1))

graph export "${sum_graph_folder}/hist/hist_`mom'_`gp'.png",as(png) replace  

}
}

* 2 groups 


foreach gp in edu_g fbetter{
foreach mom in `Moments'{

twoway (hist `mom' if `gp'==0,fcolor(gs15) lcolor("")) /// 
       (hist `mom' if `gp'==1,fcolor(ltblue) lcolor("")), ///
	   xtitle("") ///
	   ytitle("") ///
	   title("`mom'") ///
	   legend(label(1 `gp'=0) label(2 `gp'=1) col(1))

graph export "${sum_graph_folder}/hist/hist_`mom'_`gp'.png",as(png) replace  

}
}

*/

/*
**********************************
*** time series pltos by group *****
**********************************


** 2 groups 
foreach agg in mean median{
  foreach gp in fbetter{
   preserve 
   
   collapse (`agg') `Moments' sp500, by(date `gp')
   xtset `gp' date 

foreach mom in `Moments'{
keep if `mom'!=.
* moments only 

twoway (tsline `mom' if `gp'== 0,lp(solid) lwidth(thick)) ///
       (tsline `mom' if `gp'== 1,lp(dash) lwidth(thick)), ///
       xtitle("date") ///
	   ytitle("") ///
	   title("`mom' by recent household finance condition") ///
	   legend(label(1 "worse") label(2 "better") col(4))
 graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'.png",as(png) replace  
 
}
  restore
}
}


** 2 groups 
foreach agg in mean median{
  foreach gp in byear_g{
   preserve 
   
   collapse (`agg') `Moments' sp500, by(date `gp')
   xtset `gp' date 

foreach mom in `Moments'{
keep if `mom'!=.
* moments only 

twoway (tsline `mom' if `gp'== 0,lp(solid) lwidth(thick)) ///
       (tsline `mom' if `gp'== 1,lp(dash) lwidth(thick)), ///
       xtitle("date") ///
	   ytitle("") ///
	   title("`mom' by generation") ///
	   legend(label(1 "before 1980s") label(2 "after 1980s") col(4))
 graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'.png",as(png) replace  
 
 
** compute correlation coefficients 
pwcorr `mom' f2.sp500 if `gp'==0, star(0.05)
local rho_1: display %4.2f r(rho) 
pwcorr `mom' f2.sp500 if `gp'==1, star(0.05)
local rho_2: display %4.2f r(rho) 
pwcorr `mom' f2.sp500 if `gp'==2, star(0.05)
local rho_3: display %4.2f r(rho) 
pwcorr `mom' f2.sp500 if `gp'==3, star(0.05)
local rho_4: display %4.2f r(rho) 

** moments and sp500 
*** correlation with stock market by group *****

twoway (bar sp500 date if `gp'== 1,yaxis(2) fcolor(gs10)) ///
       (tsline `mom' if `gp'== 1,lp(shortdash) lwidth(thick)), ///	   
       xtitle("date") ///
	   ytitle("") ///
	   ytitle("sp500 return (%)",axis(2)) ///
	   title("`mom' by generation") ///
	   legend(label(1 "sp500 (RHS)")  label(2 "1980s") col(3)) ///
	   caption("{superscript:1950s corr =`rho_1', 1960s corr =`rho_2', 1970s corr =`rho_3',1980s corr =`rho_4',}", ///
	   justification(left) position(11) size(vlarge))
 graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'_stk.png",as(png) replace 
 
}
  restore
}
}



** 4 groups 
foreach agg in mean median{
  foreach gp in byear_g{
   preserve 
   
   collapse (`agg') `Moments' sp500, by(date `gp')
   xtset `gp' date 

foreach mom in `Moments'{
keep if `mom'!=.
* moments only 

twoway (tsline `mom' if `gp'== 0,lp(solid) lwidth(thick)) ///
       (tsline `mom' if `gp'== 1,lp(dash) lwidth(thick)) ///
	   (tsline `mom' if `gp'== 2,lp(shortdash) lwidth(thick)) ///
	   (tsline `mom' if `gp'== 3,lp(dash_dot) lwidth(thick)), ///
       xtitle("date") ///
	   ytitle("") ///
	   title("`mom' by generation") ///
	   legend(label(1 "1950s") label(2 "1960s") label(3 "1970s") label(4 "1980s")  col(4))
 graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'.png",as(png) replace  
 
 
** compute correlation coefficients 
pwcorr `mom' f11.sp500 if `gp'==0, star(0.05)
local rho_1: display %4.2f r(rho) 
pwcorr `mom' f11.sp500 if `gp'==1, star(0.05)
local rho_2: display %4.2f r(rho) 
pwcorr `mom' f11.sp500 if `gp'==2, star(0.05)
local rho_3: display %4.2f r(rho) 
pwcorr `mom' f11.sp500 if `gp'==3, star(0.05)
local rho_4: display %4.2f r(rho) 

** moments and sp500 
*** correlation with stock market by group *****

twoway (bar sp500 date if `gp'== 3,yaxis(2) fcolor(gs10)) ///
       (tsline `mom' if `gp'== 3,lp(shortdash) lwidth(thick)), ///	   
       xtitle("date") ///
	   ytitle("") ///
	   ytitle("sp500 return (%)",axis(2)) ///
	   title("`mom' by generation") ///
	   legend(label(1 "sp500 (RHS)")  label(2 "1980s") col(3)) ///
	   caption("{superscript:1950s corr =`rho_1', 1960s corr =`rho_2', 1970s corr =`rho_3',1980s corr =`rho_4',}", ///
	   justification(left) position(11) size(vlarge))
 graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'_stk.png",as(png) replace 
 
}
  restore
}
}


** 2 groups fo numeracy literacy 

foreach agg in mean median{
  foreach gp in nlit_g{
   preserve 
   
   collapse (`agg') `Moments' sp500, by(date `gp')
   xtset `gp' date 
   
foreach mom in `Moments'{
keep if `mom'!=.
twoway (tsline `mom' if `gp'== 0,lp(solid) lwidth(thick)) ///
       (tsline `mom' if `gp'== 1,lp(dash) lwidth(thick)), ///
       xtitle("date") ///
	   ytitle("") ///
	   title("`mom' by numeracy literacy") ///
	   legend(label(1 "low") label(2 "high") col(2))
 graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'.png",as(png) replace  
      
}
 restore
}
}


** 3 groups fo HH income 
foreach agg in mean median{
  foreach gp in HHinc_g{
   preserve 
   
   collapse (`agg') `Moments' sp500, by(date `gp')
   xtset `gp' date 
   
** plots for moments by group 
   
* moments only 
foreach mom in `Moments'{
keep if `mom'!=.

* moments only 
twoway (tsline `mom' if `gp'== 0,lp(solid) lwidth(thick)) ///
       (tsline `mom' if `gp'== 1,lp(dash) lwidth(thick)), ///
       xtitle("date") ///
	   ytitle("") ///
	   title("`mom' by household income") ///
	   legend(label(1 "low") label(2 "high")  col(3))
 graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'.png",as(png) replace  
 
** compute correlation coefficients 
pwcorr `mom' f2.sp500 if `gp'==0, star(0.05)
local rho_lw: display %4.2f r(rho) 
pwcorr `mom' f2.sp500 if `gp'==1, star(0.05)
local rho_md: display %4.2f r(rho) 
*pwcorr `mom' f2.sp500 if `gp'==2, star(0.05)
*local rho_hg: display %4.2f r(rho) 

** moments and sp500 
*** correlation with stock market by group *****

twoway (bar sp500 date if `gp'== 1,yaxis(2) fcolor(gs10)) ///
	   (tsline `mom' if `gp'== 1,lp(shortdash) lwidth(thick)), ///
       xtitle("date") ///
	   ytitle("") ///
	   ytitle("sp500 return (%)",axis(2)) ///
	   title("`mom' by household income") ///
	   legend(label(1 "sp500 (RHS)")  label(2 "high income") col(3)) ///
	   caption("{superscript:low corr =`rho_lw', med corr =`rho_md', high corr =`rho_hg',}", ///
	   justification(left) position(11) size(vlarge))
 graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'_stk.png",as(png) replace 
}
	  
  restore
}
}


** 3 groups fo age
foreach agg in mean median{
  foreach gp in age_g{
   preserve 
   
   collapse (`agg') `Moments' sp500, by(date `gp')
   xtset `gp' date 

 ** moments only 
foreach mom in `Moments'{
keep if `mom'!=.
twoway (tsline `mom' if `gp'== 0,lp(solid) lwidth(thick)) ///
       (tsline `mom' if `gp'== 1,lp(dash) lwidth(thick)) ///
	   (tsline `mom' if `gp'== 2,lp(shortdash) lwidth(thick)), ///
       xtitle("date") ///
	   ytitle("") ///
	   title("`mom' by age") ///
	   legend(label(1 "young") label(2 "middle-age") label(3 "old")  col(3))
	   
 graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'.png",as(png) replace  
 
 
** compute correlation coefficients 
pwcorr `mom' f2.sp500 if `gp'==0, star(0.05)
local rho_y: display %4.2f r(rho) 
pwcorr `mom' f2.sp500 if `gp'==1, star(0.05)
local rho_m: display %4.2f r(rho) 
pwcorr `mom' f2.sp500 if `gp'==2, star(0.05)
local rho_o: display %4.2f r(rho) 

** moments and sp500 
*** correlation with stock market by group *****

twoway (bar sp500 date if `gp'== 2,yaxis(2) fcolor(gs10)) ///
	   (tsline `mom' if `gp'== 2,lp(shortdash) lwidth(thick)) , ///
       xtitle("date") ///
	   ytitle("") ///
	   ytitle("sp500 return (%)",axis(2)) ///
	   title("`mom' by age") ///
	   legend(label(1 "sp500 (RHS)") label(2 "old")  col(3)) ///
	   caption("{superscript:young corr =`rho_y', middle-age corr =`rho_m', old corr =`rho_o',}", ///
	   justification(left) position(11) size(vlarge))
 graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'_stk.png",as(png) replace 
  }
 
  restore
}
}




** 2 groups of education 
foreach agg in mean median{
  foreach gp in edu_g{
   preserve 
   
   collapse (`agg') `Moments' sp500, by(date `gp')
   xtset `gp' date 
   
foreach mom in `Moments'{
keep if `mom'!=.
twoway (tsline `mom' if `gp'== 0,lp(solid) lwidth(thick)) ///
       (tsline `mom' if `gp'== 1,lp(dash) lwidth(thick)), ///
       xtitle("date") ///
	   ytitle("") ///
	   title("`mom' by education") ///
	   legend(label(1 "low") label(2 "high") col(2))
 graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'.png",as(png) replace  
      

 
** compute correlation coefficients 
pwcorr `mom' f2.sp500 if `gp'==0, star(0.05)
local rho_l: display %4.2f r(rho) 
pwcorr `mom' f2.sp500 if `gp'==1, star(0.05)
local rho_h: display %4.2f r(rho) 

** moments and sp500 
*** correlation with stock market by group *****

twoway (bar sp500 date if `gp'== 1,yaxis(2) fcolor(gs10)) ///
	   (tsline `mom' if `gp'== 1,lp(shortdash) lwidth(thick)), ///
       xtitle("date") ///
	   ytitle("") ///
	   ytitle("sp500 return (%)",axis(2)) ///
	   title("`mom' by education") ///
	   legend(label(1 "sp500 (RHS)") label(2 "high") col(3)) ///
	   caption("{superscript:low corr =`rho_l', high corr =`rho_h',}", ///
	   justification(left) position(11) size(vlarge))
     graph export "${sum_graph_folder}/ts/ts_`mom'_`gp'_`agg'_stk.png",as(png) replace 
}
  
 restore
}
}
*/

********************
** Regression ******
********************


global other_control i.gender i.Q34 Q35_1 Q35_2 Q35_3 Q35_4 Q35_5 Q35_6 
global macro_ex_var UEprobAgg Stkprob Q9_mean UEprobInd

eststo clear

foreach mom in var iqr mean{
eststo: reg inc`mom' i.age_g i.edu_g i.HHinc_g i.byear_g i.year i.state_id, robust 
eststo: reg inc`mom' i.age_g i.edu_g i.HHinc_g i.byear_g i.year i.state_id ${other_control}, robust 
eststo: reg inc`mom' i.age_g i.edu_g i.HHinc_g i.byear_g i.year i.state_id ${other_control} ${macro_ex_var}, robust 
}

esttab using "${sum_table_folder}/mom_group.csv", ///
             se r2 drop(0.age_g 0.edu_g 0.HHinc_g 0.byear_g  *.year *state_id 1.gender 1.Q34 _cons) ///
			 label replace

*****************************
** Regression Full-table (OLD)******
*******************************
			 
eststo clear

label var D6 "HH income group"
label var Q10_1 "full-time"
label var Q10_2 "part-time"
label var Q36 "education"
	
foreach mom in mean var iqr rmean rvar{
eststo: reg inc`mom' i.Q10_2 i.Q12new i.month, vce(cl ID)
eststo: reg inc`mom' i.Q10_2 i.Q12new i.D6 i.month,vce(cl ID)
eststo: reg inc`mom' i.Q10_2 i.Q12new i.D6 Q4new Q13new Q6new i.month,vce(cl ID)
eststo: reg inc`mom' i.Q36 i.age_g i.month,vce(cl ID)
}

/*
foreach mom in Mean Var Skew Kurt{
eststo: xtreg Inc`mom' i.Q10_1 i.Q10_2 i.Q12new, fe robust 
eststo: xtreg Inc`mom' i.Q10_1 i.Q10_2 i.Q12new i.Q47, fe robust 
eststo: xtreg Inc`mom' i.Q10_1 i.Q10_2 i.Q12new i.Q47 Q4new Q13new Q6new, fe robust 
}
*/

esttab using "${sum_table_folder}/mom_ind_reg.csv", ///
             se r2 drop(0.age_g 1.Q36 0.Q10_2 1.Q12new 1.D6 *.month _cons) ///
			 label replace
*/	
log close 
