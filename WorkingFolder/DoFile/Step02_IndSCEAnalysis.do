clear
global mainfolder "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/Graphs/ind"
global sum_table_folder "${mainfolder}/Tables"

cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/indSCE_log",replace


***************************
**  Clean and Merge Data **
***************************

use "${folder}/SCE/IncExpSCEProbIndM",clear 

duplicates report year month userid


************************************
**  merge with estimated moments **
***********************************

* IncExpSCEDstIndM is the output from ../Pythoncode/DoDensityEst.ipynb
merge 1:1 userid date using "${folder}/SCE/IncExpSCEDstIndM.dta", keep(master using) 
rename _merge IndEst_merge

rename userid ID 

*******************************
**  Set Panel Data Structure **
*******************************

xtset ID date
sort ID year month 

*******************************
**  Summary Statistics of SCE **
*******************************

tabstat ID,s(count) by(date) column(statistics)

*****************************************
****  Renaming so that more consistent **
*****************************************

rename Q24_mean incmean
rename Q24_var incvar
rename Q24_iqr inciqr
rename IncSkew incskew 
rename Q24_rmean rincmean
rename Q24_rvar rincvar

*rename Q36 educ
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

*******************************
**  Generate Variables       **
*******************************

gen Incmean = .
gen Incvar = .
gen Inciqr = .

gen Incmean_ch = .
gen Incvar_ch = .
gen Inciqr_ch = .


gen Inc_mom = .
gen Inc_mom_ch = .

*** take the log

foreach mom in var{
	gen linc`mom'= log(inc`mom') 
	gen lrinc`mom' = log(rinc`mom') 
}


foreach mom in iqr{
	gen linc`mom'= log(inc`mom') 
}

label var linciqr "log perceived iqr"
label var lincvar "log perceived risk"
label var lrincvar "log perceived risk (real)"
	
************************************************
** spending decisions and perceived risks **
************************************************
eststo clear
xtset ID date

label var UEprobAgg "UE expecation"

eststo: reg spending lrincvar
eststo: areg spending lrincvar, a(date)
eststo: areg spending lrincvar, a(ID)
eststo: areg spending lrincvar i.year, a(ID)
eststo: areg spending lincvar, a(ID)
eststo: areg spending UEprobAgg, a(date)

esttab using "${sum_table_folder}/ind/spending_reg_fe.csv", label mtitles se r2 ///
 drop(_cons *.year) replace
eststo clear

************************************************
** pesistence of the individual perceived risks **
************************************************

eststo clear

foreach mom in mean var{
    xtset ID date
	replace Inc_mom = inc`mom' 
    *replace Incmom_ch = Inc`mom'- l1.Inc`mom'

	eststo `mom'1: reg Inc_mom l(1/3).Inc_mom, vce(cluster date)
	eststo `mom'2: reg Inc_mom l(1/6).Inc_mom, vce(cluster date)
	eststo `mom'3: reg Inc_mom l(1/8).Inc_mom, vce(cluster date)
	
	replace Inc_mom = rinc`mom'
	eststo r`mom'1: reg Inc_mom l(1/3).Inc_mom, vce(cluster date)
	eststo r`mom'2: reg Inc_mom l(1/6).Inc_mom, vce(cluster date)
	eststo r`mom'3: reg Inc_mom l(1/8).Inc_mom, vce(cluster date)
 }
 
  
esttab using "${sum_table_folder}/ind/autoregIndM.csv", mtitles se  r2 replace
eststo clear


log close 
