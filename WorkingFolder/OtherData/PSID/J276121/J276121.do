#delimit ;
*  PSID DATA CENTER *****************************************************
   JOBID            : 276121                            
   DATA_DOMAIN      : IND                               
   USER_WHERE       : NULL                              
   FILE_TYPE        : All Individuals Data              
   OUTPUT_DATA_TYPE : ASCII                             
   STATEMENTS       : do                                
   CODEBOOK_TYPE    : PDF                               
   N_OF_VARIABLES   : 6                                 
   N_OF_OBSERVATIONS: 18230                             
   MAX_REC_LENGTH   : 14                                
   DATE & TIME      : April 29, 2020 @ 15:36:42
*************************************************************************
;

infix
      ER30000              1 - 1           ER30001              2 - 5           ER30002              6 - 8     
      ER30003              9 - 9           ER30011             10 - 10          ER30012             11 - 14    
using /Users/Myworld/Dropbox/IncExpProject/WorkingFolder/OtherData/PSID/J276121/J276121.txt, clear 
;
label variable ER30000       "RELEASE NUMBER"                           ;
label variable ER30001       "1968 INTERVIEW NUMBER"                    ;
label variable ER30002       "PERSON NUMBER                         68" ;
label variable ER30003       "RELATIONSHIP TO HEAD                  68" ;
label variable ER30011       "TYPE OF INCOME                        68" ;
label variable ER30012       "MONEY INCOME IND                      68" ;
