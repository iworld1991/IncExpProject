*******************************************************************
*  Stata "do-file" file with labels and missing data specifications
*  Created by ddltox on Mar 20, 2020  (Fri 03:51 PM EDT)
*  DDL source file: "/z/sca-v2/sda/public/htdocs/tmpdir/AAWkzUxp.txt".
*
*  Note that the data dictionary is given at the end of this file.
*  Put the dictionary into a separate file (by editing this file).
*  Then specify below the name of the dictionary file.
*
*  DDL file gives the following dataset description:
*    Records per case: 1
*    Record length:    176
*******************************************************************

clear

label data "Surveys of Consumers"

#delimit ;
label define PAGO      1 "Better now" 3 "Same" 5 "Worse now" 8 "DK" 9 "NA" ;
label define PAGOR1    0 "No change and no pro-con reason given" 
                       10 "FAV Better pay" 
                       11 "FAV Higher income from self-employment or property" 
                       12 "FAV More work, hence more income" 
                       13 "FAV Increased contributions from outside FU" 
                       14 "FAV Lower prices" 
                       15 "FAV Lower taxes; low or unchanged taxes" 
                       16 "FAV Decreased expenses" 
                       18 "FAV Higher interest rates; tight credit" 
                       19 "FAV Better asset position" 
                       20 "FAV Debt, interest or debt payments low or lower" 
                       21 "FAV Change in family composition - higher income or better off" 
                       23 "FAV Good times, no recession (not codeable above)" 
                       27 "FAV Other reasons for making FU better off" 
                       38 "FAV Reference to government economic policy" 
                       39 "FAV Income tax refund" 50 "UNFAV Lower pay" 
                       51 "UNFAV Lower income from self-employment or property" 
                       52 "UNFAV Less work, hence less income" 
                       53 "UNFAV Decreased/Unchanged contributions from outside FU" 
                       54 "UNFAV High(er) prices" 
                       55 "UNFAV Higher interest rates; tight credit" 
                       56 "UNFAV High, higher taxes (except 57)" 
                       57 "UNFAV Income taxes" 
                       58 "UNFAV Increased expenses; more people to be supported by FU" 
                       59 "UNFAV Worse asset position" 60 "UNFAV Debt" 
                       61 "UNFAV Change in family composition - lower income or worse off" 
                       63 "UNFAV Bad times, recession (not codeable above)" 
                       64 "UNFAV Strike(s)-- not codeable in 52" 
                       67 "UNFAV Other reasons for making FU worse off" 
                       78 "UNFAV Reference to government economic policy" 
                       98 "DK" 99 "NA" ;
label define PAGOR2    0 "No second mention" 10 "FAV Better pay" 
                       11 "FAV Higher income from self-employment or property" 
                       12 "FAV More work, hence more income" 
                       13 "FAV Increased contributions from outside FU" 
                       14 "FAV Lower prices" 
                       15 "FAV Lower taxes; low or unchanged taxes" 
                       16 "FAV Decreased expenses" 
                       18 "FAV Higher interest rates; tight credit" 
                       19 "FAV Better asset position" 
                       20 "FAV Debt, interest or debt payments low or lower" 
                       21 "FAV Change in family composition - higher income or better off" 
                       23 "FAV Good times, no recession (not codeable above)" 
                       27 "FAV Other reasons for making FU better off" 
                       38 "FAV Reference to government economic policy" 
                       39 "FAV Income tax refund" 50 "UNFAV Lower pay" 
                       51 "UNFAV Lower income from self-employment or property" 
                       52 "UNFAV Less work, hence less income" 
                       53 "UNFAV Decreased/Unchanged contributions from outside FU" 
                       54 "UNFAV High(er) prices" 
                       55 "UNFAV Higher interest rates; tight credit" 
                       56 "UNFAV High, higher taxes (except 57)" 
                       57 "UNFAV Income taxes" 
                       58 "UNFAV Increased expenses; more people to be supported by FU" 
                       59 "UNFAV Worse asset position" 60 "UNFAV Debt" 
                       61 "UNFAV Change in family composition - lower income or worse off" 
                       63 "UNFAV Bad times, recession (not codeable above)" 
                       64 "UNFAV Strike(s)-- not codeable in 52" 
                       67 "UNFAV Other reasons for making FU worse off" 
                       78 "UNFAV Reference to government economic policy" 
                       98 "DK" 99 "NA" ;
label define PAGO5     1 "Better now" 3 "Same" 5 "Worse now" 8 "DK" 9 "NA" ;
label define PEXP      1 "Will be better off" 3 "Same" 5 "Will be worse off" 
                       8 "DK" 9 "NA" ;
label define PEXP5     1 "Will be better off" 3 "Same" 5 "Will be worse off" 
                       8 "DK" 9 "NA" ;
label define INEXQ1    1 "Higher" 3 "About the same" 5 "Lower" 8 "DK" 9 "NA" ;
label define INEXQ2    95 "95% or more" 98 "DK" 99 "NA" ;
label define INEX      -97 "DK how much down" 96 "DK how much up" 
                       98 "DK whether up or down" 99 "NA" ;
label define RINC      1 "Income up more than prices" 
                       3 "Income up same as prices" 
                       5 "Income up less than prices" 8 "DK" 9 "NA" ;
label define DUR       1 "Good" 3 "Pro-con" 5 "Bad" 8 "DK" 9 "NA" ;
label define DURRN1    0 "No second mention" 
                       10 "FAV Interest rates won't get any lower" 
                       11 "FAV Prices are low(er), prices reasonably stable" 
                       12 "FAV Good buys available, sales, discounts" 
                       13 "FAV Prices are going up, future uncertainty" 
                       14 "FAV Prices won't get any lower" 
                       15 "FAV Lower down payment" 
                       16 "FAV Interest rates low" 
                       17 "FAV Credit easy to get, easy money" 
                       18 "FAV Interest rates are going up, credit tighter" 
                       19 "FAV Low taxes, tax changes" 
                       21 "FAV People can afford to buy now, have money to spend" 
                       23 "FAV Buying makes for good times, prosperity" 
                       31 "FAV Supply adequate, no shortages now" 
                       32 "FAV Quality is good/better/may get worse" 
                       33 "FAV New models have improvements/new features" 
                       34 "FAV Good selection, variety" 
                       41 "FAV Seasonal references only" 
                       42 "FAV R says if you need it, good time as any" 
                       43 "FAV Low sales won't last, will pick up soon" 
                       47 "FAV Other good reasons" 
                       49 "FAV Economic policy, references to gov't/president" 
                       50 "UNFAV Interest rates won't get any lower" 
                       51 "UNFAV Prices are too high, prices going up" 
                       52 "UNFAV Seller's market, few sales or discounts" 
                       53 "UNFAV Prices will fall later, will come down" 
                       54 "UNFAV Debt or credit is bad" 
                       55 "UNFAV Larger/higher down payment required" 
                       56 "UNFAV Interest rates high/going up" 
                       57 "UNFAV Credit/financing hard to get; tight money" 
                       58 "UNFAV Interest rates will fall later" 
                       59 "UNFAV Taxes high, going higher" 
                       61 "UNFAV People can't afford to buy now" 
                       62 "UNFAV People should save money" 
                       63 "UNFAV Buying contributes to inflation, makes for bad times" 
                       65 "UNFAV Energy crisis; shortages of fuels" 
                       71 "UNFAV Supply inadequate, poor selection" 
                       72 "UNFAV Quality is poor, may improve later" 
                       73 "UNFAV Poor designs; unattractive styling" 
                       81 "UNFAV R mentions only seasonal factors" 
                       82 "UNFAV International references" 
                       87 "UNFAV Other reasons why now is a bad time to buy" 
                       89 "UNFAV Economic policy, references to gov't/president" 
                       98 "DK" 99 "NA" ;
label define DURRN2    0 "No second mention" 
                       10 "FAV Interest rates won't get any lower" 
                       11 "FAV Prices are low(er), prices reasonably stable" 
                       12 "FAV Good buys available, sales, discounts" 
                       13 "FAV Prices are going up, future uncertainty" 
                       14 "FAV Prices won't get any lower" 
                       15 "FAV Lower down payment" 
                       16 "FAV Interest rates low" 
                       17 "FAV Credit easy to get, easy money" 
                       18 "FAV Interest rates are going up, credit tighter" 
                       19 "FAV Low taxes, tax changes" 
                       21 "FAV People can afford to buy now, have money to spend" 
                       23 "FAV Buying makes for good times, prosperity" 
                       31 "FAV Supply adequate, no shortages now" 
                       32 "FAV Quality is good/better/may get worse" 
                       33 "FAV New models have improvements/new features" 
                       34 "FAV Good selection, variety" 
                       41 "FAV Seasonal references only" 
                       42 "FAV R says if you need it, good time as any" 
                       43 "FAV Low sales won't last, will pick up soon" 
                       47 "FAV Other good reasons" 
                       49 "FAV Economic policy, references to gov't/president" 
                       50 "UNFAV Interest rates won't get any lower" 
                       51 "UNFAV Prices are too high, prices going up" 
                       52 "UNFAV Seller's market, few sales or discounts" 
                       53 "UNFAV Prices will fall later, will come down" 
                       54 "UNFAV Debt or credit is bad" 
                       55 "UNFAV Larger/higher down payment required" 
                       56 "UNFAV Interest rates high/going up" 
                       57 "UNFAV Credit/financing hard to get; tight money" 
                       58 "UNFAV Interest rates will fall later" 
                       59 "UNFAV Taxes high, going higher" 
                       61 "UNFAV People can't afford to buy now" 
                       62 "UNFAV People should save money" 
                       63 "UNFAV Buying contributes to inflation, makes for bad times" 
                       65 "UNFAV Energy crisis; shortages of fuels" 
                       71 "UNFAV Supply inadequate, poor selection" 
                       72 "UNFAV Quality is poor, may improve later" 
                       73 "UNFAV Poor designs; unattractive styling" 
                       81 "UNFAV R mentions only seasonal factors" 
                       82 "UNFAV International references" 
                       87 "UNFAV Other reasons why now is a bad time to buy" 
                       89 "UNFAV Economic policy, references to gov't/president" 
                       98 "DK" 99 "NA" ;
label define HOM       1 "Good" 3 "Pro-con" 5 "Bad" 8 "DK" 9 "NA" ;
label define HOMRN1    0 "No second mention" 
                       10 "FAV Interest rate won't get any lower" 
                       11 "FAV Prices are low/stable/not too high" 
                       12 "FAV Good buys available" 
                       13 "FAV Prices are going up" 
                       14 "FAV Prices won't get any lower" 
                       15 "FAV Lower down payment" 
                       16 "FAV Interest rates are low" 
                       17 "FAV Credit easy to get, easy money" 
                       18 "FAV Credit will be tighter later" 
                       19 "FAV Lower taxes, taxes higher later" 
                       21 "FAV People can afford to buy now" 
                       23 "FAV Buying makes for good times, prosperity" 
                       27 "FAV Other references to employment and purchasing power" 
                       31 "FAV Supply adequate, no shortages now" 
                       32 "FAV Quality is good, better, may get worse" 
                       33 "FAV New models have improvements, new features" 
                       34 "FAV Good selection, variety" 
                       41 "FAV Seasonal references only" 
                       42 "FAV R only says: if you need it this is a good time" 
                       43 "FAV Low sales won't last, will pick up soon" 
                       44 "FAV Renting is unfavorable b/c high rents, shortage" 
                       45 "FAV Owning is always a good idea, renting is a bad idea" 
                       46 "FAV Capital appreciation: buying is a good investment" 
                       47 "FAV Other good reasons (misc.)" 
                       48 "FAV Variable mortgage rate" 
                       49 "FAV Economic policy, references to gov't/president" 
                       50 "UNFAV Interest rates won't get any lower" 
                       51 "UNFAV Prices are too high, houses cost too much" 
                       52 "UNFAV Seller's market, few sales or discounts" 
                       53 "UNFAV Prices will fall later, will come down" 
                       54 "UNFAV Debt or credit bad (NA why)" 
                       55 "UNFAV Higher/larger down payment required" 
                       56 "UNFAV Interest rate too high, will go up" 
                       57 "UNFAV Credit hard to get, financing difficult" 
                       58 "UNFAV Interest rates will come down later" 
                       59 "UNFAV Tax increase, property taxes too high" 
                       61 "UNFAV People can't afford to buy now, times are bad" 
                       62 "UNFAV People should save money, uncertain of future" 
                       63 "UNFAV Buying contributes to inflation/makes bad times" 
                       65 "UNFAV Energy crisis, shortage of fuels" 
                       71 "UNFAV Supply inadequate, few houses on market" 
                       72 "UNFAV Quality is poor, quality may improve" 
                       73 "UNFAV Poor designs, unattractive styling" 
                       81 "UNFAV R mentions only seasonal factors" 
                       82 "UNFAV Difficult to get rid of present house" 
                       83 "UNFAV Better return on alternative investments" 
                       84 "UNFAV Renting favorable b/c of low rents" 
                       85 "UNFAV Renting is always better than owning" 
                       86 "UNFAV Capital depreciation, buying is bad investment" 
                       87 "UNFAV Other reasons why now is a bad time to buy" 
                       88 "UNFAV Variable mortgage rate" 
                       89 "UNFAV Economic policy, references to gov't/president" 
                       98 "DK" 99 "NA" ;
label define HOMRN2    0 "No second mention" 
                       10 "FAV Interest rate won't get any lower" 
                       11 "FAV Prices are low/stable/not too high" 
                       12 "FAV Good buys available" 
                       13 "FAV Prices are going up" 
                       14 "FAV Prices won't get any lower" 
                       15 "FAV Lower down payment" 
                       16 "FAV Interest rates are low" 
                       17 "FAV Credit easy to get, easy money" 
                       18 "FAV Credit will be tighter later" 
                       19 "FAV Lower taxes, taxes higher later" 
                       21 "FAV People can afford to buy now" 
                       23 "FAV Buying makes for good times, prosperity" 
                       27 "FAV Other references to employment and purchasing power" 
                       31 "FAV Supply adequate, no shortages now" 
                       32 "FAV Quality is good, better, may get worse" 
                       33 "FAV New models have improvements, new features" 
                       34 "FAV Good selection, variety" 
                       41 "FAV Seasonal references only" 
                       42 "FAV R only says: if you need it this is a good time" 
                       43 "FAV Low sales won't last, will pick up soon" 
                       44 "FAV Renting is unfavorable b/c high rents, shortage" 
                       45 "FAV Owning is always a good idea, renting is a bad idea" 
                       46 "FAV Capital appreciation: buying is a good investment" 
                       47 "FAV Other good reasons (misc.)" 
                       48 "FAV Variable mortgage rate" 
                       49 "FAV Economic policy, references to gov't/president" 
                       50 "UNFAV Interest rates won't get any lower" 
                       51 "UNFAV Prices are too high, houses cost too much" 
                       52 "UNFAV Seller's market, few sales or discounts" 
                       53 "UNFAV Prices will fall later, will come down" 
                       54 "UNFAV Debt or credit bad (NA why)" 
                       55 "UNFAV Higher/larger down payment required" 
                       56 "UNFAV Interest rate too high, will go up" 
                       57 "UNFAV Credit hard to get, financing difficult" 
                       58 "UNFAV Interest rates will come down later" 
                       59 "UNFAV Tax increase, property taxes too high" 
                       61 "UNFAV People can't afford to buy now, times are bad" 
                       62 "UNFAV People should save money, uncertain of future" 
                       63 "UNFAV Buying contributes to inflation/makes bad times" 
                       65 "UNFAV Energy crisis, shortage of fuels" 
                       71 "UNFAV Supply inadequate, few houses on market" 
                       72 "UNFAV Quality is poor, quality may improve" 
                       73 "UNFAV Poor designs, unattractive styling" 
                       81 "UNFAV R mentions only seasonal factors" 
                       82 "UNFAV Difficult to get rid of present house" 
                       83 "UNFAV Better return on alternative investments" 
                       84 "UNFAV Renting favorable b/c of low rents" 
                       85 "UNFAV Renting is always better than owning" 
                       86 "UNFAV Capital depreciation, buying is bad investment" 
                       87 "UNFAV Other reasons why now is a bad time to buy" 
                       88 "UNFAV Variable mortgage rate" 
                       89 "UNFAV Economic policy, references to gov't/president" 
                       98 "DK" 99 "NA" ;
label define SHOM      1 "GOOD" 3 "PRO-CON" 5 "BAD" 8 "DK" 9 "NA" ;
label define SHOMRN1   0 "No second mention" 
                       10 "FAV Interest rate won't get any lower (not codeable elsewhere)" 
                       11 "FAV Prices are high/higher/won't get any lower" 
                       12 "FAV Seller's market (under-supply of houses)" 
                       13 "FAV Prices going down; sell before prices lower" 
                       14 "FAV Prices won't get any higher (not codeable 13)" 
                       15 "FAV Lower down payment" 
                       16 "FAV Interest rates are low (now)" 
                       17 "FAV Credit easy to get; easy money, NA if 15, 16, 17, or 18" 
                       18 "FAV Credit will be tighter later; interest rates will go up" 
                       19 "FAV Lower taxes; taxes will be higher later" 
                       21 "FAV People can afford to buy now" 
                       23 "FAV Buying makes for good times/prosperity/high employment" 
                       31 "FAV Supply inadequate, shortages now; may be shortages later" 
                       33 "FAV Good time for existing homes, costs more to build new ones" 
                       41 "FAV Seasonal references only" 
                       42 "FAV R only says: If need to sell/need money this is good time" 
                       44 "FAV Can use cash/capital for other investments" 
                       45 "FAV Better to sell now, value of home may decline" 
                       46 "FAV Capital appreciation: value of houses increased; good profits now" 
                       47 "FAV Other good reasons (miscellaneous)" 
                       48 "FAV Variable mortgage rate" 
                       49 "FAV Economic policy; references to gov't/new president" 
                       50 "UNFAV Interest rates won't get any lower (not codeable elsewhere)" 
                       51 "UNFAV Prices are low/lower" 
                       52 "UNFAV Buyer's market; difficult to find buyers;" 
                       53 "UNFAV Prices will rise later; future uncertainty about prices" 
                       54 "UNFAV Interest rates low/lower" 
                       55 "UNFAV Higher/Larger down payment required" 
                       56 "UNFAV Interest rate too high; will go up" 
                       57 "UNFAV Credit hard to get; financing difficult; pt system; tight money" 
                       58 "UNFAV Interest rates will come down later; credit easier later" 
                       59 "UNFAV Tax increase; (property) taxes too high; going higher" 
                       61 "UNFAV People can't afford to buy now; recession; inflation" 
                       62 "UNFAV People should save money; future uncertain; bad times ahead" 
                       63 "UNFAV Buying contributes to inflation/makes for bad times" 
                       65 "UNFAV Energy crisis; shortages of fuels; high price of utilities" 
                       71 "UNFAV Supply adequate; (no reference to influence on prices/deals)" 
                       73 "UNFAV Bad time for older homes; people want newer homes" 
                       81 "UNFAV R mentions only seasonal factors" 
                       84 "UNFAV Home is good\better investment" 
                       85 "UNFAV Rents are too high" 
                       86 "UNFAV Capital depreciation: would lose money if sold now" 
                       87 "UNFAV Other reasons why now is a bad time to sell" 
                       88 "UNFAV Variable mortgage rate" 
                       89 "UNFAV Economic policy; references to government/new president" 
                       98 "DK" 99 "NA" ;
label define SHOMRN2   0 "No second mention" 
                       10 "FAV Interest rate won't get any lower (not codeable elsewhere)" 
                       11 "FAV Prices are high/higher/won't get any lower" 
                       12 "FAV Seller's market (under-supply of houses)" 
                       13 "FAV Prices going down; sell before prices lower" 
                       14 "FAV Prices won't get any higher (not codeable 13)" 
                       15 "FAV Lower down payment" 
                       16 "FAV Interest rates are low (now)" 
                       17 "FAV Credit easy to get; easy money, NA if 15, 16, 17, or 18" 
                       18 "FAV Credit will be tighter later; interest rates will go up" 
                       19 "FAV Lower taxes; taxes will be higher later" 
                       21 "FAV People can afford to buy now" 
                       23 "FAV Buying makes for good times/prosperity/high employment" 
                       31 "FAV Supply inadequate, shortages now; may be shortages later" 
                       33 "FAV Good time for existing homes, costs more to build new ones" 
                       41 "FAV Seasonal references only" 
                       42 "FAV R only says: If need to sell/need money this is good time" 
                       44 "FAV Can use cash/capital for other investments" 
                       45 "FAV Better to sell now, value of home may decline" 
                       46 "FAV Capital appreciation: value of houses increased; good profits now" 
                       47 "FAV Other good reasons (miscellaneous)" 
                       48 "FAV Variable mortgage rate" 
                       49 "FAV Economic policy; references to gov't/new president" 
                       50 "UNFAV Interest rates won't get any lower (not codeable elsewhere)" 
                       51 "UNFAV Prices are low/lower" 
                       52 "UNFAV Buyer's market; difficult to find buyers;" 
                       53 "UNFAV Prices will rise later; future uncertainty about prices" 
                       54 "UNFAV Interest rates low/lower" 
                       55 "UNFAV Higher/Larger down payment required" 
                       56 "UNFAV Interest rate too high; will go up" 
                       57 "UNFAV Credit hard to get; financing difficult; pt system; tight money" 
                       58 "UNFAV Interest rates will come down later; credit easier later" 
                       59 "UNFAV Tax increase; (property) taxes too high; going higher" 
                       61 "UNFAV People can't afford to buy now; recession; inflation" 
                       62 "UNFAV People should save money; future uncertain; bad times ahead" 
                       63 "UNFAV Buying contributes to inflation/makes for bad times" 
                       65 "UNFAV Energy crisis; shortages of fuels; high price of utilities" 
                       71 "UNFAV Supply adequate; (no reference to influence on prices/deals)" 
                       73 "UNFAV Bad time for older homes; people want newer homes" 
                       81 "UNFAV R mentions only seasonal factors" 
                       84 "UNFAV Home is good\better investment" 
                       85 "UNFAV Rents are too high" 
                       86 "UNFAV Capital depreciation: would lose money if sold now" 
                       87 "UNFAV Other reasons why now is a bad time to sell" 
                       88 "UNFAV Variable mortgage rate" 
                       89 "UNFAV Economic policy; references to government/new president" 
                       98 "DK" 99 "NA" ;
label define CAR       1 "Good" 3 "Pro-con" 5 "Bad" 8 "DK" 9 "NA" ;
label define CARRN1    0 "No second mention" 
                       10 "FAV Interest rates won't get any lower" 
                       11 "FAV Prices are low/lower/stable/not too high" 
                       12 "FAV Good buys available; sales, discounts" 
                       13 "FAV Prices are going up, buy before prices higher" 
                       14 "FAV Prices won't get any lower" 
                       15 "FAV Lower down payment" 
                       16 "FAV Interest rates low" 
                       17 "FAV Credit easy to get; easy money" 
                       18 "FAV Interest rates are going higher" 
                       19 "FAV Taxes low; will be higher" 
                       20 "FAV Rebate/Bonus program" 
                       21 "FAV People can afford to buy now; purchasing power up" 
                       23 "FAV Buying makes for good times, prosperity" 
                       25 "FAV Energy crisis lessened, availability of gas" 
                       30 "FAV New cars get better mileage, due to gasahol" 
                       31 "FAV Supply adequate, no shortages now" 
                       32 "FAV Quality is good/better/may get worse" 
                       33 "FAV New models have improvements; new features" 
                       34 "FAV Great variety of models and sizes to choose from" 
                       35 "FAV (New) Small (economy) cars" 
                       36 "FAV Safety; new models are safer" 
                       37 "FAV Safety devices will be on and that's bad" 
                       38 "FAV Anti-pollution devices (will be on, good)" 
                       39 "FAV Anti-pollution devices (will be on, bad)" 
                       40 "FAV Strikes: labor problems, union demands" 
                       41 "FAV Seasonal reference only" 
                       42 "FAV R says: if you need it, good time as any" 
                       43 "FAV Low sales won't last, will pick up soon" 
                       44 "FAV NA whether 36 or 38, or both" 
                       45 "FAV NA whether 37 or 39, or both" 
                       46 "FAV New models are little changed from old" 
                       47 "FAV Other good reasons (misc.)" 
                       49 "FAV Economic policy, references to gov't/president" 
                       50 "UNFAV Interest rates won't get any lower" 
                       51 "UNFAV Prices are (too) high, prices are going up" 
                       52 "UNFAV Seller's market; few sales or discounts" 
                       53 "UNFAV Prices will fall later, are falling" 
                       54 "UNFAV Debt or credit is bad (NA why)" 
                       55 "UNFAV Larger/higher down payment required" 
                       56 "UNFAV Interest rates are high, will go up" 
                       57 "UNFAV Credit hard to get, tight money" 
                       58 "UNFAV Interest rates will fall later" 
                       59 "UNFAV Taxes high, going higher" 
                       60 "UNFAV Because rebate/bonus program will be over" 
                       61 "UNFAV People can't afford to buy now, times bad" 
                       62 "UNFAV People should save money, bad times ahead" 
                       63 "UNFAV Buying contributes to inflation, makes bad times" 
                       65 "UNFAV Energy crisis, gas shortage, price of gas" 
                       67 "UNFAV Environmental/ecology reasons; pollution" 
                       70 "UNFAV Poor mileage (including due to gasahol)" 
                       71 "UNFAV Supply inadequate; few cars on market" 
                       72 "UNFAV Quality is poor; quality better later" 
                       73 "UNFAV Poor designs; unattractive styling" 
                       74 "UNFAV New types of cars will be intro soon" 
                       75 "UNFAV New smaller cars" 
                       76 "UNFAV Safety; later models will be safer" 
                       77 "UNFAV Too many safe items (expensive, unneeded)" 
                       78 "UNFAV Later models will pollute less" 
                       79 "UNFAV Anti-pollution devices (will be on, bad)" 
                       80 "UNFAV Strikes; labor problems" 
                       81 "UNFAV R mentions only seasonal factors" 
                       82 "UNFAV Imported car market; int'l references" 
                       83 "UNFAV High sales can't last, change is due" 
                       84 "UNFAV NA whether 76, or 78, or both" 
                       85 "UNFAV NA whether 77, or 79, or both" 
                       86 "UNFAV Poor performance, not clear why" 
                       87 "UNFAV Other reasons why now is a bad time to buy" 
                       88 "UNFAV Cost of insurance" 
                       89 "UNFAV Economic policy, references to gov't/president" 
                       90 "UNFAV Good for imported cars, bad for domestic" 
                       91 "UNFAV Good time for new car, bad time for used" 
                       92 "UNFAV Good time for used cars, bad time for new" 
                       93 "UNFAV Depends on whether new or used" 
                       94 "UNFAV Good time for small cars, bad for big cars" 
                       95 "UNFAV Good time for big cars, bad for small cars" 
                       96 "UNFAV Good for domestic cars, bad for imported" 
                       98 "DK" 99 "NA" ;
label define CARRN2    0 "No second mention" 
                       10 "FAV Interest rates won't get any lower" 
                       11 "FAV Prices are low/lower/stable/not too high" 
                       12 "FAV Good buys available; sales, discounts" 
                       13 "FAV Prices are going up, buy before prices higher" 
                       14 "FAV Prices won't get any lower" 
                       15 "FAV Lower down payment" 
                       16 "FAV Interest rates low" 
                       17 "FAV Credit easy to get; easy money" 
                       18 "FAV Interest rates are going higher" 
                       19 "FAV Taxes low; will be higher" 
                       20 "FAV Rebate/Bonus program" 
                       21 "FAV People can afford to buy now; purchasing power up" 
                       23 "FAV Buying makes for good times, prosperity" 
                       25 "FAV Energy crisis lessened, availability of gas" 
                       30 "FAV New cars get better mileage, due to gasahol" 
                       31 "FAV Supply adequate, no shortages now" 
                       32 "FAV Quality is good/better/may get worse" 
                       33 "FAV New models have improvements; new features" 
                       34 "FAV Great variety of models and sizes to choose from" 
                       35 "FAV (New) Small (economy) cars" 
                       36 "FAV Safety; new models are safer" 
                       37 "FAV Safety devices will be on and that's bad" 
                       38 "FAV Anti-pollution devices (will be on, good)" 
                       39 "FAV Anti-pollution devices (will be on, bad)" 
                       40 "FAV Strikes: labor problems, union demands" 
                       41 "FAV Seasonal reference only" 
                       42 "FAV R says: if you need it, good time as any" 
                       43 "FAV Low sales won't last, will pick up soon" 
                       44 "FAV NA whether 36 or 38, or both" 
                       45 "FAV NA whether 37 or 39, or both" 
                       46 "FAV New models are little changed from old" 
                       47 "FAV Other good reasons (misc.)" 
                       49 "FAV Economic policy, references to gov't/president" 
                       50 "UNFAV Interest rates won't get any lower" 
                       51 "UNFAV Prices are (too) high, prices are going up" 
                       52 "UNFAV Seller's market; few sales or discounts" 
                       53 "UNFAV Prices will fall later, are falling" 
                       54 "UNFAV Debt or credit is bad (NA why)" 
                       55 "UNFAV Larger/higher down payment required" 
                       56 "UNFAV Interest rates are high, will go up" 
                       57 "UNFAV Credit hard to get, tight money" 
                       58 "UNFAV Interest rates will fall later" 
                       59 "UNFAV Taxes high, going higher" 
                       60 "UNFAV Because rebate/bonus program will be over" 
                       61 "UNFAV People can't afford to buy now, times bad" 
                       62 "UNFAV People should save money, bad times ahead" 
                       63 "UNFAV Buying contributes to inflation, makes bad times" 
                       65 "UNFAV Energy crisis, gas shortage, price of gas" 
                       67 "UNFAV Environmental/ecology reasons; pollution" 
                       70 "UNFAV Poor mileage (including due to gasahol)" 
                       71 "UNFAV Supply inadequate; few cars on market" 
                       72 "UNFAV Quality is poor; quality better later" 
                       73 "UNFAV Poor designs; unattractive styling" 
                       74 "UNFAV New types of cars will be intro soon" 
                       75 "UNFAV New smaller cars" 
                       76 "UNFAV Safety; later models will be safer" 
                       77 "UNFAV Too many safe items (expensive, unneeded)" 
                       78 "UNFAV Later models will pollute less" 
                       79 "UNFAV Anti-pollution devices (will be on, bad)" 
                       80 "UNFAV Strikes; labor problems" 
                       81 "UNFAV R mentions only seasonal factors" 
                       82 "UNFAV Imported car market; int'l references" 
                       83 "UNFAV High sales can't last, change is due" 
                       84 "UNFAV NA whether 76, or 78, or both" 
                       85 "UNFAV NA whether 77, or 79, or both" 
                       86 "UNFAV Poor performance, not clear why" 
                       87 "UNFAV Other reasons why now is a bad time to buy" 
                       88 "UNFAV Cost of insurance" 
                       89 "UNFAV Economic policy, references to gov't/president" 
                       90 "UNFAV Good for imported cars, bad for domestic" 
                       91 "UNFAV Good time for new car, bad time for used" 
                       92 "UNFAV Good time for used cars, bad time for new" 
                       93 "UNFAV Depends on whether new or used" 
                       94 "UNFAV Good time for small cars, bad for big cars" 
                       95 "UNFAV Good time for big cars, bad for small cars" 
                       96 "UNFAV Good for domestic cars, bad for imported" 
                       98 "DK" 99 "NA" ;
label define INCQFM    
                       1 "Income in open format (before 1990)/ Asked open format, answered open format" 
                       2 "Asked open format, answered bracketed format: coded to midpoint" 
                       3 "Income in bracket format (before 1990)/ Asked bracketed, answered bracketed: coded to midpoint" ;
label define YTL10     1 "Bottom 10%" 5 "Top 90%" ;
label define YTL90     1 "Top 10%" 5 "Bottom 90%" ;
label define YTL50     1 "Bottom 50%" 5 "Top 50%" ;
label define YTL5      1 "Bottom 20%" 2 "21-40%" 3 "41-60%" 4 "61-80%" 
                       5 "Top 20%" ;
label define YTL4      1 "Bottom 25%" 2 "26-50%" 3 "51-75%" 4 "Top 25%" ;
label define YTL3      1 "Bottom 33%" 2 "Middle 33%" 3 "Top 33%" ;
label define HOMEOWN   1 "Owns or is buying" 2 "Rent" 99 "NA" ;
label define HOMEQFM   1 "Asked open format, answered open format" 
                       2 "Asked open format, answered bracketed format: coded to midpoint" 
                       3 "Asked bracketed question, answered bracketed format: coded to midpoint" ;
label define HTL10     1 "Bottom 10%" 5 "Top 90%" ;
label define HTL90     1 "Top 10%" 5 "Bottom 90%" ;
label define HTL50     1 "Bottom 50%" 5 "Top 50%" ;
label define HTL5      1 "Bottom 20%" 2 "21-40%" 3 "41-60%" 4 "61-80%" 
                       5 "Top 20%" ;
label define HTL4      1 "Bottom 25%" 2 "26-50%" 3 "51-75%" 4 "Top 25%" ;
label define HTL3      1 "Bottom 33%" 2 "Middle 33%" 3 "Top 33%" ;
label define HOMEVAL   1 "Increased in value" 3 "Same" 5 "Decreased in value" 
                       8 "DK" 9 "NA" ;
label define HOMPX1Q1  1 "Increase" 3 "About the same" 5 "Decrease" 8 "DK" 
                       9 "NA" ;
label define HOMPX1Q2  998 "DK" 999 "NA" ;
label define HOMPX1    -997 "DK how much down" 996 "DK how much up" 
                       998 "DK whether up or down" 999 "NA" ;
label define HOMPX5Q1  1 "Increase" 3 "Remain about the same" 5 "Decrease" 
                       8 "DK" 9 "NA" ;
label define HOMPX5Q2  998 "DK" 999 "NA" ;
label define HOMPX5    -997 "DK how much down" 996 "DK how much up" 
                       998 "DK whether up or down" 999 "NA" ;
label define INVEST    1 "Yes" 5 "No" ;
label define INVQFM    1 "Asked open format, answered open format" 
                       2 "Asked open format, answered bracketed format: coded to midpoint" 
                       3 "Asked bracketed format, answered bracketed format: coded to midpoint" ;
label define STL10     1 "Bottom 10%" 5 "Top 90%" ;
label define STL90     1 "Top 10%" 5 "Top 90%" ;
label define STL50     1 "Bottom 50%" 5 "Top 50%" ;
label define STL5      1 "Bottom 20%" 2 "21-40%" 3 "41-60%" 4 "61-80%" 
                       5 "Top 20%" ;
label define STL4      1 "Bottom 25%" 2 "26-50%" 3 "51-75%" 4 "Top 25%" ;
label define STL3      1 "Bottom 33%" 2 "Middle 33%" 3 "Top 33%" ;
label define AGE       97 "97 or older" ;
label define BIRTHM    1 "January" 2 "February" 3 "March" 4 "April" 5 "May" 
                       6 "June" 7 "July" 8 "August" 9 "September" 
                       10 "October" 11 "November" 12 "December" ;
label define REGION    1 "West" 2 "North Central" 3 "Northeast" 4 "South" ;
label define SEX       1 "Male" 2 "Female" ;
label define MARRY     1 "Married/partner" 2 "Separated" 3 "Divorced" 
                       4 "Widowed" 5 "Never married" ;
label define NUMKID    0 "None" ;
label define EDUC      1 "Grade 0-8 no hs diploma" 
                       2 "Grade 9-12 no hs diploma" 
                       3 "Grade 0-12 w/ hs diploma" 
                       4 "Grade 13-17 no col degree" 
                       5 "Grade 13-16 w/ col degree" 
                       6 "Grade 17 W/ col degree" ;
label define ECLGRD    1 "Yes" 5 "No" 8 "DK" 9 "NA" ;
label define EHSGRD    1 "Yes" 5 "No" 8 "DK" 9 "NA" ;
label define EGRADE    98 "DK" 99 "NA" ;
label define PINC      998 "DK" 999 "NA" ;
label define PINC2     996 "Volunteered 'No personal income'" 998 "DK" 
                       999 "NA" ;
label define PJOB      998 "DK" 999 "NA" ;
label define PSSA      998 "DK" 999 "NA" ;
label define PCRY      1 "Gone up" 3 "Same" 5 "Gone down" 8 "DK" 9 "NA" ;
label define PSTK      998 "DK" 999 "NA" ;


#delimit cr

*******************************************************************
infile using dictionary
* Replace 'X' with the name of the dictionary file. 
*
* The contents of the dictionary are given at the end of this file.
* Put the dictionary into a separate file (by editing this file).
* Then specify here the name of the dictionary file.
*******************************************************************
* The md, min and max specifications were translated 
* into the following "REPLACE...IF" statements:

replace PAGO = . if (PAGO >= 8 ) 
replace PAGOR1 = . if (PAGOR1 >= 98 ) 
replace PAGOR2 = . if (PAGOR2 >= 98 ) 
replace PAGO5 = . if (PAGO5 >= 8 ) 
replace PEXP = . if (PEXP >= 8 ) 
replace PEXP5 = . if (PEXP5 >= 8 ) 
replace INEXQ1 = . if (INEXQ1 >= 8 ) 
replace INEXQ2 = . if (INEXQ2 >= 98 ) 
replace INEX = . if (INEX == -97)
replace INEX = . if (INEX >= 96 ) 
replace RINC = . if (RINC >= 8 ) 
replace DUR = . if (DUR >= 8 ) 
replace DURRN1 = . if (DURRN1 >= 98 ) 
replace DURRN2 = . if (DURRN2 >= 98 ) 
replace HOM = . if (HOM >= 8 ) 
replace HOMRN1 = . if (HOMRN1 >= 98 ) 
replace HOMRN2 = . if (HOMRN2 >= 98 ) 
replace SHOM = . if (SHOM >= 8 ) 
replace CAR = . if (CAR >= 8 ) 
replace CARRN1 = . if (CARRN1 >= 98 ) 
replace CARRN2 = . if (CARRN2 >= 98 ) 
replace HOMEOWN = . if (HOMEOWN >= 99 ) 
replace HOMEVAL = . if (HOMEVAL >= 8 ) 
replace HOMPX1Q1 = . if (HOMPX1Q1 >= 8 ) 
replace HOMPX1Q2 = . if (HOMPX1Q2 >= 998 ) 
replace HOMPX1 = . if (HOMPX1 == -997)
replace HOMPX1 = . if (HOMPX1 >= 996 ) 
replace HOMPX5Q1 = . if (HOMPX5Q1 >= 8 ) 
replace HOMPX5Q2 = . if (HOMPX5Q2 >= 998 ) 
replace HOMPX5 = . if (HOMPX5 == -997)
replace HOMPX5 = . if (HOMPX5 >= 996 ) 
replace ECLGRD = . if (ECLGRD >= 8 ) 
replace EHSGRD = . if (EHSGRD >= 8 ) 
replace EGRADE = . if (EGRADE >= 98 ) 
replace PINC = . if (PINC >= 998.00 ) 
replace PINC2 = . if (PINC2 >= 998.00 ) 
replace PJOB = . if (PJOB >= 998.00 ) 
replace PSSA = . if (PSSA >= 998.00 ) 
replace PCRY = . if (PCRY >= 8 ) 
replace PSTK = . if (PSTK >= 998.00 ) 

