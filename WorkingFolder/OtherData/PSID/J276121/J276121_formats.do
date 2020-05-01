
label define ER30000L  ///
       1 "Release number 1, February 2019"  ///
       2 "Release number 2, May 2019"  ///
       3 "Release number 3, August 2019"  ///
       4 "Release number 4, February 2020"

label define ER30003L  ///
       1 "Head"  ///
       2 `"Wife/"Wife""'  ///
       3 "Son or daughter"  ///
       4 "Brother or sister"  ///
       5 "Father or mother"  ///
       6 "Grandchild, niece, nephew, other relatives under 18"  ///
       7 "Other, including in-laws, other adult relatives"  ///
       8 "Husband or Wife of Head who moved out or died in the year prior to the 1968 interview"  ///
       9 "NA"  ///
       0 "Individual from core sample who was born or moved in after the 1968 interview; individual from Immigrant or Latino samples (ER30001=3001-3511, 4001-4462,7001-9308)"

label define ER30011L  ///
       1 "Labor income only"  ///
       2 "Transfer income only"  ///
       3 "Asset income only"  ///
       4 "Combination of labor, transfer, and/or asset"  ///
       9 "NA; DK"  ///
       0 "Inap.:  no income; born or moved in after the 1968 interview or individual from Immigrant or Latino samples (ER30003=0); under 14 years old in 1968 (ER30004=1-13)"

label values ER30000    ER30000L
label values ER30003    ER30003L
label values ER30011    ER30011L
