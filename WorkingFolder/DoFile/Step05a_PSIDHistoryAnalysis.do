clear
import excel "/Users/Myworld/Dropbox/IncExpProject/WorkingFolder/Tables/psid/psid_history_vol.xls", sheet("Sheet1") firstrow
destring year cohort rmse, force replace
graph bar rmse if year==2017, over(cohort) 
hist rmse
gen rmseqrt = sqrt(rmse)
hist rmseqrt
gen tenure = year-cohort 
graph bar (mean) rmse, over(tenure)
