clear
global mainfolder "/Users/Myworld/Dropbox/IncomeExp/WorkingFolder"
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

rename Q24_mean SCE_Mean
rename Q24_var SCE_Var

*******************************
**  Generate Variables       **
*******************************

gen IncExp_Mean = .
gen IncExp_Var = .

gen IncExp_Mean_ch = .
gen IncExp_Var_ch = .

gen IncExp_Mean_rv = .
gen IncExp_Var_rv = .

************************************************
** Auto Regression of the Individual Moments  **
************************************************

eststo clear

foreach mom in Mean Var{
   foreach var in SCE{
    replace IncExp_`mom' = `var'_`mom'
	xtset ID date
    replace IncExp_`mom'_ch = IncExp_`mom'-l1.IncExp_`mom'

	eststo `var'_`mom'lvl: reg IncExp_`mom' l(3/5).IncExp_`mom', vce(cluster date)
    eststo `var'_`mom'diff: reg IncExp_`mom'_ch l(3/5).IncExp_`mom'_ch, vce(cluster date)
  }
}
esttab using "${sum_table_folder}/ind/autoregSCEIndM.csv", mtitles se  r2 replace
eststo clear




*******************************************************
***  Weak test on changes of forecst and uncertainty **
*******************************************************

** Generate central tendency measures

foreach var in SCE{
foreach mom in Mean{
   egen `var'_`mom'_ct50 = pctile(`var'_`mom'),p(50) by(date)
   label var `var'_`mom'_ct50 "Median 1-year-ahead income growth"
}
}

eststo clear

foreach var in SCE{
  foreach mom in Mean{
     replace IncExp_`mom'_ch =  `var'_`mom' - l1.`var'_`mom'
	 eststo `var'`mom'diff0: reg IncExp_`mom'_ch, vce(cluster date)
     eststo `var'`mom'diff1: reg IncExp_`mom'_ch l1.IncExp_`mom'_ch, vce(cluster date)
	 capture eststo `var'`mom'diff2: reg  IncExp_`mom'_ch l(1/3).IncExp_`mom'_ch, vce(cluster date)
	 capture eststo `var'`mom'diff3: reg  IncExp_`mom'_ch l(1/6).IncExp_`mom'_ch, vce(cluster date)
 }
}

foreach var in SCE{
  foreach mom in Var{
     replace IncExp_`mom'_ch =  `var'_`mom' - l1.`var'_`mom'
	 eststo `var'`mom'diff0: reg IncExp_`mom'_ch, vce(cluster date) 
     eststo `var'`mom'diff1: reg IncExp_`mom'_ch l1.IncExp_`mom'_ch, vce(cluster date) 
	 eststo `var'`mom'diff2: reg  IncExp_`mom'_ch l(1/3).IncExp_`mom'_ch, vce(cluster date) 
	 capture eststo `var'`mom'diff3: reg  IncExp_`mom'_ch l(1/6).IncExp_`mom'_ch, vce(cluster date)
 }
}

esttab using "${sum_table_folder}/ind/ChEfficiencySCEIndQ.csv", mtitles b(%8.3f) se(%8.3f) scalars(N r2) sfmt(%8.3f %8.3f %8.3f) replace



/*
** There is no revision in income growth in SCE.  
** Use 2-year inflation forecast 10 months ago and 1-year forecast now

***************************************************
*** Revision Efficiency Test Using Mean Revision **
***************************************************


eststo clear

foreach var in SCE{
  foreach mom in Mean{
     replace IncExp_`mom'_rv =  `var'_`mom' - l10.`var'_`mom'1
	 eststo `var'`mom'rvlv0: reg IncExp_`mom'_rv, vce(cluster date)
     eststo `var'`mom'rvlv1: reg IncExp_`mom'_rv l1.IncExp_`mom'_rv `var'_`mom'_ct50, vce(cluster date)
	 *eststo `var'`mom'rvlv2: reg  InfExp_`mom'_rv l(1/2).InfExp_`mom'_rv `var'_`mom'_ct50, vce(cluster date)
	 *eststo `var'`mom'rvlv3: reg  InfExp_`mom'_rv l(1/3).InfExp_`mom'_rv `var'_`mom'_ct50, vce(cluster date)
 }
}

foreach var in SCE{
  foreach mom in Var{
     replace IncExp_`mom'_rv =  `var'_`mom' - l10.`var'_`mom'1
	 eststo `var'`mom'rvlv0: reg IncExp_`mom'_rv, vce(cluster date) 
     eststo `var'`mom'rvlv1: reg IncExp_`mom'_rv l1.IncExp_`mom'_rv, vce(cluster date) 
	 *eststo `var'`mom'rvlv2: reg  InfExp_`mom'_rv l(1/2).InfExp_`mom'_rv, vce(cluster date) 
	 *eststo `var'`mom'rvlv3: reg  InfExp_`mom'_rv l(1/3).InfExp_`mom'_rv, vce(cluster date)
 }
}

esttab using "${sum_table_folder}/ind/RVEfficiencySCEIndQ.csv", mtitles b(%8.3f) se(%8.3f) scalars(N r2) sfmt(%8.3f %8.3f %8.3f) replace
*/

log close 
