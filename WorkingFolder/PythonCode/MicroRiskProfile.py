# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### How do the perceived income moments correlate with other individual variables 
#
# - this notebook runs regressions to inspects the covariants of individual perceived income moments
#   - individual demogrpahics, level of household income, education, etc.
#   - job-types, part-time vs full-time, selfemployment, etc. 
#   - other expectations: probability of unemployment, other job-related expectations 
# - it examiens both nominal and real income growth 
#

# ###  1. Loading and cleaning data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import statsmodels.api as sm 
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from statsmodels.iolib.summary2 import summary_col

"""
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf','png','jpg')
#plt.rcParams['savefig.dpi'] = 75

#plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
#plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
"""

pd.options.display.float_format = '{:,.2f}'.format

dataset = pd.read_stata('../SurveyData/SCE/IncExpSCEProbIndM.dta')   

# + {"code_folding": [0]}
## panel data 

#dataset.index = dataset[['date','userid']]

# + {"code_folding": []}
## variable list by catogrories 

vars_id = ['userid','date']

moms_nom = ['Q24_mean','Q24_var']

moms_real = ['Q24_rmean','Q24_rvar']

vars_demog = ['D6']   ## level of income, 11 groups 

vars_job = ['Q10_1',  # full-time 
            'Q10_2',  # part-time
            'Q12new'] ## =1 worked for others; = 2 self-employment 

vars_demog_sub = ['Q32',  ## age 
                  'Q33',  ## gender 
                  'Q36']  ## education (1-8 low to high, 9 other)

## these variables are only available for a sub sample 

vars_empexp = ['Q13new']  ## probability of unemployment 

vars_macroexp = ['Q6new',  ## stock market going up 
                 'Q4new']  ## UE goes up 

# + {"code_folding": []}
## subselect variables 

vars_all_reg_long = (vars_id+moms_nom + moms_real + vars_job + 
                     vars_demog + vars_demog_sub + 
                     vars_empexp + vars_macroexp)

## select dataset 

SCEM = dataset[vars_all_reg_long]
SCEM.describe(include='all')

# +
## renaming 

SCEM = SCEM.rename(columns={'Q24_mean': 'incexp',
                           'Q24_var': 'incvar',
                           'Q24_iqr': 'inciqr',
                           'Q24_rmean':'rincexp',
                           'Q24_rvar': 'rincvar'})

SCEM = SCEM.rename(columns = {'D6':'HHinc',
                              'Q13new':'UEprobInd',
                              'Q6new':'Stkprob',
                              'Q4new':'UEprobAgg',
                              'Q10_1':'fulltime',
                              'Q10_2':'parttime',
                              'Q12new':'selfemp',
                              'Q32':'age',
                              'Q33':'gender',
                              'Q36':'educ'})

SCEM.columns
# -

# ### 2. Correlation pattern 

SCEM.dtypes

for col in ['HHinc','age','educ']:
    SCEM[col] = SCEM[col].astype('int64',errors = 'ignore')

# + {"code_folding": [0]}
inc_grp = {1:"10k",
           2:'20k',
           3:'30k',
           4:'40k',
           5:'50k',
           6:'60k',
           7:'75k',
           8:'100k',
           9:'150k',
           10:'200k',
           11:'200k+'}

cleanup_nums = {'parttime': {0: 'no', 1: 'yes'},
                'fulltime': {0: 'no', 1: 'yes'},
                'selfemp':{1: 'no', 2: 'yes'},
                'gender':{1:'male',2:'female'}}
SCEM.replace(cleanup_nums,
             inplace = True)

# +
## categorical variables 

vars_cat = ['HHinc','fulltime','parttime','selfemp','gender','educ','userid','date']

for var in vars_cat:
    SCEM[var] = pd.Categorical(SCEM[var],ordered = False)

# +
#pp = sns.pairplot(SCEM)

# + {"code_folding": []}
sns.heatmap(SCEM.corr(), annot=True)
# -

# ###  3. Histograms

# + {"code_folding": []}
## by income group 

fontsize = 80
figsize = (100,50)

"""
for mom in ['incvar','rincvar']:
    for gp in ['HHinc','educ','gender']:
        plt.style.use('seaborn-poster')
        SCEM.boxplot(column=[mom],
                     figsize = figsize,
                     by = gp,
                     patch_artist = True,
                     fontsize = fontsize)
        plt.xlabel(gp,
                   fontsize = fontsize)
        plt.ylabel(mom,
                   fontsize = fontsize)
        plt.ylim(0,40)
        plt.suptitle('')
        plt.title(mom, fontsize= fontsize)
        plt.savefig('../Graphs/ind/boxplot'+str(mom)+'_'+str(gp)+'.jpg')
        
"""

# + {"code_folding": [5, 6]}
fontsize = 80
figsize = (100,40)
plt.style.use('ggplot')


for gp in ['HHinc','educ','gender']:
    bp = SCEM.boxplot(column=['incvar','rincvar'],
                      figsize = figsize,
                      by = gp,
                      patch_artist = True,
                      fontsize = fontsize,
                      layout=(1, 2),
                      rot =45,
                      return_type='dict')
    
    #plt.title(mom, fontsize= fontsize)
    ## adjust width of lines
    [[item.set_linewidth(4) for item in bp[key]['boxes']] for key in bp.keys()]
    [[item.set_linewidth(40) for item in bp[key]['fliers']] for key in bp.keys()]
    [[item.set_linewidth(40) for item in bp[key]['medians']] for key in bp.keys()]
    [[item.set_linewidth(40) for item in bp[key]['means']] for key in bp.keys()]
    #[[item.set_linewidth(40) for item in bp[key]['whiskers']] for key in bp.keys()]
    [[item.set_linewidth(40) for item in bp[key]['caps']] for key in bp.keys()]
    
    ## adjust color 
    #[[item.set_markerfacecolor('r') for item in bp[key]['means']] for key in bp.keys()]
    #[[item.set_color('k') for item in bp[key]['whiskers']] for key in bp.keys()]
    
    
    plt.xlabel(gp,fontsize = fontsize)
    plt.ylabel('var', fontsize = fontsize)
    plt.ylim(0,50)
    plt.suptitle('')
    ## save figure 
    plt.savefig('../Graphs/ind/boxplot'+'_'+str(gp)+'.jpg')

# + {"code_folding": [9]}
gplist = ['HHinc','educ','gender']
momlist = ['incvar','rincvar']
incg_lb = list(inc_grp.values())


## plot 

fig,axes = plt.subplots(3,2,figsize =(20,20))

for i in range(3):
    for j in range(2):
        gp = gplist[i]
        mom = momlist[j]
        if gplist[i] =='HHinc':
            bp = sns.boxplot(x = gp,
                            y = mom,
                            data = SCEM, 
                            color = 'skyblue',
                            ax = axes[i,j],
                            whis = True,
                            showfliers = False)
        else:
            bp = sns.boxplot(x = gp,
                             y = mom,
                             data = SCEM,
                             color = 'skyblue',
                             ax = axes[i,j],
                             showfliers = False)
            
        # settings 
        bp.set_xlabel(gp,fontsize = 20)
        #bp.xtick_label(fontsize=20)
        bp.set_ylabel(mom,fontsize = 20)
        
plt.savefig('../Graphs/ind/boxplot.jpg')
# -

# ###  4. Regressions

# +
## preps 

dep_list =  ['incvar','rincvar'] 
dep_list2 =['incexp','rincexp']
indep_list_ct = ['UEprobInd','UEprobInd','Stkprob']
indep_list_dc = ['HHinc','selfemp','fulltime']


# + {"code_folding": [7]}
## full-table for risks  

rs_list = {}  ## list to store results 
nb_spc = 4  ## number of specifications 

for i,mom in enumerate(dep_list):
    ## model 1 
    model = smf.ols(formula = str(mom)
                    +'~ C(parttime)+C(selfemp)',
                    data = SCEM)
    rs_list[nb_spc*i] = model.fit()
    
    ## model 2
    ct_str = '+'.join([var for var in indep_list_ct])
    model2 = smf.ols(formula = str(mom)
                    +'~ C(parttime)+C(selfemp) + '
                     + ct_str,
                    data = SCEM)
    rs_list[nb_spc*i+1] = model2.fit()
    
    ## model 3 
    model3 = smf.ols(formula = str(mom)
                    +'~ C(parttime) + C(selfemp) + C(HHinc) +'
                     + ct_str,
                    data = SCEM)
    rs_list[nb_spc*i+2] = model3.fit()
    
    ## model 4 
    model4 = smf.ols(formula = str(mom)
                    +'~ C(gender)+ C(educ)',
                    data = SCEM)
    rs_list[nb_spc*i+3] = model4.fit()
    
    
rs_names = [rs_list[i] for i in range(len(rs_list))]

dfoutput = summary_col(rs_names,
                        float_format='%0.2f',
                        stars = True,
                        info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                                  'R2':lambda x: "{:.2f}".format(x.rsquared)})
dfoutput.title = 'Perceived Income Risks'
print(dfoutput)

# +
## output tables 

beginningtex = """
\\begin{table}[ht]
\\centering
\\begin{adjustbox}{width={0.9\\textwidth},totalheight={\\textheight}}
\\begin{threeparttable}
\\caption{Perceived Income Risks and Individual Characteristics}
\\label{micro_reg}"""

endtex = """\\begin{tablenotes}\item Standard errors are clustered by household. *** p$<$0.001, ** p$<$0.01 and * p$<$0.05. 
\item This table reports regression results of perceived income risks (incvar for nominal, rincvar for real) on household specific variables. HHinc: household income group ranges from lowests (=1, less than \$10k/year) to the heightst (=11, greater than \$200k/year). Education, educ ranges from the lowest (=1, less than high school) to the highest (=9).
\\end{tablenotes}
\\end{threeparttable}
\\end{adjustbox}
\\end{table}"""


# + {"code_folding": [3]}
## relabel rows 


def CatRename(table):
    relabels = {}
    rows = [idx for idx in table.index if ')[T.' in idx]
    for i in range(len(rows)):
        string = rows[i]
        var = string.split('C(')[1].split(')[T')[0]
        val = string.split('[T.')[1].split(']')[0]
        if '.0' in val:
            val = val.split('.0')[0]
        else:
            val = val 
        relabels[rows[i]] = var + '=' + str(val)
    table = table.rename(index = relabels)
    return table 

table = CatRename(dfoutput.tables[0])


# +
## excluding rows that are not to be exported 

to_drop = ['Intercept']

## need to also drop rows reporting the stadard deviations as well 
rows_below = []
for var in to_drop:
    row_idx = list(table.index).index(var)
    #print(row_idx)
    rows_below.append(row_idx) 
    
tb = table.drop(index = to_drop)

# + {"code_folding": []}
## write to latex 
f = open('../Tables/latex/micro_reg.tex', 'w')
f.write(beginningtex)
tb_ltx = tb.to_latex().replace('lllllllll','ccccccccc')   # hard coded here 
#print(tb)
f.write(tb_ltx)
f.write(endtex)
f.close()
# + {"code_folding": []}
## full-table for expected growth, appendix 

## full-table for risks  

rs_list = {}  ## list to store results 
nb_spc = 4  ## number of specifications 

for i,mom in enumerate(dep_list2):
    ## model 1 
    model = smf.ols(formula = str(mom)
                    +'~ C(parttime)+C(selfemp)',
                    data = SCEM)
    rs_list[nb_spc*i] = model.fit()
    
    ## model 2
    ct_str = '+'.join([var for var in indep_list_ct])
    model2 = smf.ols(formula = str(mom)
                    +'~ C(parttime)+C(selfemp) + '
                     + ct_str,
                    data = SCEM)
    rs_list[nb_spc*i+1] = model2.fit()
    
    ## model 3 
    model3 = smf.ols(formula = str(mom)
                    +'~ C(parttime) + C(selfemp) + C(HHinc) +'
                     + ct_str,
                    data = SCEM)
    rs_list[nb_spc*i+2] = model3.fit()
    
    ## model 4 
    model4 = smf.ols(formula = str(mom)
                    +'~ C(gender)+ C(educ)',
                    data = SCEM)
    rs_list[nb_spc*i+3] = model4.fit()
    
    
rs_names2 = [rs_list[i] for i in range(len(rs_list))]

dfoutput2 = summary_col(rs_names2,
                        float_format='%0.2f',
                        stars = True,
                        info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                                  'R2':lambda x: "{:.2f}".format(x.rsquared)})
dfoutput2.title = 'Perceived Income Growth'
print(dfoutput2)

## relabel 
table = CatRename(dfoutput2.tables[0])


## drop 
to_drop = ['Intercept']

## need to also drop rows reporting the stadard deviations as well 
rows_below = []
for var in to_drop:
    row_idx = list(table.index).index(var)
    #print(row_idx)
    rows_below.append(row_idx) 
    
tb = table.drop(index = to_drop)

## latex setting 


beginningtex = """
\\begin{table}[ht]
\\centering
\\begin{adjustbox}{width={0.9\\textwidth},totalheight={\\textheight}}
\\begin{threeparttable}
\\caption{Perceived Income Growth and Individual Characteristics}
\\label{micro_reg_exp}"""

endtex = """\\begin{tablenotes}\item Standard errors are clustered by household. *** p$<$0.001, ** p$<$0.01 and * p$<$0.05. 
\item This table reports regression results of perceived labor income(incexp for nominal, rincexp for real) growth on household specific variables. HHinc: household income group ranges from lowests (=1, less than \$10k/year) to the heightst (=11, greater than \$200k/year). Education, educ ranges from the lowest (=1, less than high school) to the highest (=9).
\\end{tablenotes}
\\end{threeparttable}
\\end{adjustbox}
\\end{table}"""


## write to latex 
f = open('../Tables/latex/micro_reg_exp.tex', 'w')
f.write(beginningtex)
tb_ltx = tb.to_latex().replace('lllllllll','ccccccccc')   # hard coded here 
#print(tb)
f.write(tb_ltx)
f.write(endtex)
f.close()
# -


