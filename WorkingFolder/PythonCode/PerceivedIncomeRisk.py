# -*- coding: utf-8 -*-
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

# # Introduction
#
#
# Income risks matter for both individual behaviors and aggregate outcomes. With identical expected income and homogeneous risk preferences, different degrees of risks lead to different saving/consumption and portfolio choices. This is well understood in models in which agents are inter-temporally risk-averse, or prudent, and the risks associated with future marginal utility motivate precautionary motives. Since it is widely known from the empirical research that idiosyncratic income risks are at most partially insured (<cite data-cite="blundell_consumption_2008">(Blundell, et al. 2013)</cite>) or because of borrowing constraints, such behavioral regularities equipped with market incompleteness leads to ex-post unequal wealth distribution and different degrees of marginal propensity to consume (MPC). This has important implications for the transmission of macroeconomic policies. 
#
# One important assumption prevailing in macroeconomic models with uninsured risks is that agents have a perfect understanding of the income risks. Under the assumption, economists typically estimate the income process based on micro income data and then treat the estimates as the true model parameters known by the agents making decisions in the model. But, as the mounting evidence that people form expectations in ways deviating from full-information rationality, leading to perennial heterogeneity in economic expectations held by micro agents, this assumption seems to be too stringent. To the extent that agents make decisions based on their *respective* perceptions, understanding the *perceived* income risk profile and its correlation structure with other macro variables are the keys to explaining their behavior patterns.
#
# This paper constructs a subjective model of learning that features agents' imperfect understanding of the size as well as the nature of income risks. In terms of the size, the imperfect understanding is modeled as a lack of knowledge of the true model parameter of an assumed income process. Agents form their best guess about the parameters by learning from the past experienced income of their own as well as others. Such experience-based learning mechanisms engenders the perceived risks to be dependent on age, generation, and macroeconomic history. At the same time, the imperfect understanding of the nature of risks is captured by assuming that individuals do not understand if the income risks are commonly shared aggregate shock or idiosyncratic ones. They learn about the model based on a subjective determination of the nature of the past shocks, which is called *attribution* in this paper. With different subjective attributions, agents arrive at different degrees of model parameter uncertainty, thus different perceptions about income risks. 
#
# In a general setting within the model, I show that a higher degree of external attribution, i.e. perceiving the shocks to be common ones instead of idiosyncratic ones, leads to higher risk perceptions. This introduces a well-specified channel through which imperfect understanding of the nature of shocks leads to differences in risk perceptions. The intuition for this is straightforward. As an econometrician would perfectly understand, learning comes from variations in the sample either cross-sectionally or over time. As agents subjectively perceived correlation of her own income shocks and others' to be higher, the uncertainty associated with the parameter estimate is naturally higher simply because the variation seen cross-sectionally from the sample useful to learning is reduced. This effect is particularly important when we consider second moments or higher in that the parameter uncertainty does not enter the parameter estimate itself. 
#
# Incorporating attribution in learning allows it possible to explore the implications of possible mischaracterization of experienced shocks. Among various possible deviations, I explore a particular kind attribution error which is reminiscent of the "self-serving bias" in social psychology. In particular, it assumes that people have a tendency to external (internal) attribution in the face of negative (positive) experiences. By allowing the subjective correlation to be a function of the recent experience such as income change, the model neatly captures this psychological tendency. What is interesting is that such a state-dependence of attribution in learning may help explain why the average perceived risk is lower for the high-income groups than the low-income ones. In the presence of aggregate risks, it also generates counter-cyclical patterns of the average perceived risks, i.e. bad times are associated with high subjective risks. 
#
# Empirically, the paper sheds light on the perceptions of income risks by utilizing the recently available density forecasts of labor income surveyed by New York Fed's Survey of Consumer Expectation (SCE). What is special about this survey is that agents are asked to provide histogram-type forecasts of their earning growth over the next 12 months together with a set of expectational questions about the macroeconomy. When the individual density forecast is available, a parametric density estimation can be made to obtain the individual-specific subjective distribution. And higher moments reflecting the perceived income risks such as variance, as well as the asymmetry of the distribution such as skewness allow me to directly characterize the perceived risk profile without relying on external estimates from cross-sectional microdata. This provides the first-hand measured perceptions on income risks that are truly relevant to individual decisions.
# Perceived income risks exhibits a number of important patterns that are consistent with the predictions of experience-based learning with subjective attribution. 
#
# - Higher experienced volatility is associated with higher perceived income risks. This helps explain why perceived risks differ systematically across different generations, who have experienced different histories of the income shocks. Besides, perceived risks declines with one's age.
#
# - Perceived income risks have a non-monotonic correlation with the current income, which can be best described as a skewed U shape. Perceived risk decreases with current income over the most range of income values follwed by a uppick in perceived risks for high-income group. 
#  
# - Perceived income risks are counter-cyclical with the labor market conditions or broadly business cycles. I found that average perceived income risks by U.S. earners are negatively correlated with the current labor market tightness measured by wage growth and unemployment rate. Besides, earners in states with higher unemployment rates and low wage growth also perceive income risks to be higher. This bears similarities but important difference with a few previous studies that document the counter-cyclicality of income risks estimated by cross-sectional microdata. (<cite data-cite="guvenen2014nature">(Guvenen, et al. 2014)</cite>, <cite data-cite="catherine_countercyclical_2019">(Catherine, 2019)</cite>) 
#
# These patterns suggest that individuals have a roughly good yet imperfect understanding of their income risks. Good, in the sense that subjective perceptions are broadly consistent with the realization of cross-sectional income patterns. This is attained in my model because agents learn from past experiences, roughly as econometricians do. In contrast, subjective perceptions are imperfect in that bounded rationality prevents people from knowing about the true income process perfectly, which even hardworking economists equipped with different advanced econometrical techniques and larger sample of income data cannot easily claim to have. 
#
# As illustrated by many empirical work of testing the rationality in expectation formation, it is admittedly challenging to separately account for the differences in perceptions driven by the "truth" and the part driven by the pure subjective heterogeneity. The most straightforward way seems to be to treat econometrician's external estimates of the income process as the proxy to the truth, for which the subjective surveys are compared. But this approach implicitly assumes that econometricians correctly specify the model of the income process and ignores the possible superior information that is available only to the people in the sample but not to econometricians. The model built in this paper reconciles both possibilities.  
#  
# Finally, the subjective learning model will be incorporated into an otherwise standard life-cycle consumption/saving model with uninsured idiosyncratic and aggregate risks. Experience-based learning makes income expectations and risks state-dependent when making dynamically optimal decisions at each point of the time. In particular, higher perceived risks will induce more precautionary saving behaviors. If this perceived risk is state-dependent on recent income changes, it will potentially shift the distribution of MPCs along income deciles, therefore, amplify the channels aggregate demand responses to shocks. 
#
#      
# ##  Related literature
#
# This paper is relevant to four lines of literature. First, the idea of this paper echoes with an old problem in the consumption insurance literature: 'insurance or information' (<cite data-cite="pistaferri_superior_2001">Pistaferri, 2001</cite>, <cite data-cite="kaufmann_disentangling_2009">Kaufmann and Pistaferri, 2009</cite>,<cite data-cite="meghir2011earnings">Meghir et al. 2011</cite>). In any empirical tests of consumption insurance or consumption response to income, there is always a worry that what is interpreted as the shock has actually already entered the agents' information set or exactly the opposite. For instance, the notion of excessive sensitivity, namely households consumption highly responsive to anticipated income shock, maybe simply because agents have not incorporated the recently realized shocks that econometricians assume so (<cite data-cite="flavin_excess_1988">Flavin,1988</cite>). Also, recently, in the New York Fed [blog](https://libertystreeteconomics.newyorkfed.org/2017/11/understanding-permanent-and-temporary-income-shocks.html), the authors followed a similar approach to decompose the permanent and transitory shocks. My paper shares a similar spirit with these studies in the sense that I try to tackle the identification problem in the same approach: directly using the expectation data and explicitly controlling what are truly conditional expectations of the agents making the decision. This helps economists avoid making assumptions on what is exactly in the agents' information set. What differentiates my work from other authors is that I focus on higher moments, i.e. income risks and skewness by utilizing the recently available density forecasts of labor income. Previous work only focuses on the sizes of the realized shocks and estimates the variance of the shocks using cross-sectional distribution, while my paper directly studies the individual specific variance of these shocks perceived by different individuals. This will become clear in Section \ref{perceived-income-process-in-progress}. 
#
# Second, this paper is inspired by an old but recently reviving interest in studying consumption/saving behaviors in models incorporating imperfect expectations and perceptions. For instance, <cite data-cite="rozsypal_overpersistence_2017">(Rozsypal and Schlafmann, 2017)</cite> found that households' expectation of income exhibits an over-persistent bias using both expected and realized household income from Michigan household survey. The paper also shows that incorporating such bias affects the aggregate consumption function by distorting the cross-sectional distributions of marginal propensity to consume(MPCs) across the population. <cite data-cite="carroll_sticky_2018">(Carroll et al. 2018)</cite> reconciles the low micro-MPC and high macro-MPCs by introducing to the model an information rigidity of households in learning about macro news while being updated about micro news. <cite data-cite="lian2019imperfect">(Lian, 2019)</cite> shows that an imperfect perception of wealth accounts for such phenomenon as excess sensitivity to current income and higher MPCs out of wealth than current income and so forth. My paper has a similar flavor to all of these works by exploring the behavioral implications of households' perceptive imperfection. The novelty of my paper lies in the primary focus on the implications of heterogeneity in perceived higher moments such as risks and skewness. Various theories of expectation formation have different predictions about the cross-sectional and dynamic patterns of perceived risks. I examine these predictions in this paper.   
#
# This paper also contributes to the literature studying expectation formation using subjective surveys. There has been a long list of "irrational expectation" theories developed in recent decades on how agents deviate from full-information rationality benchmark, such as sticky expectation, noisy signal extraction, least-square learning, etc. Also, empirical work has been devoted to testing these theories in a comparable manner (<cite data-cite="coibion2012can">(Coibion and Gorodnichenko, 2012)</cite>, <cite data-cite="fuhrer2018intrinsic">(Fuhrer, 2018)</cite>). But it is fair to say that thus far, relatively little work has been done on individual variables such as labor income, which may well be more relevant to individual economic decisions. Therefore, understanding expectation formation of the individual variables, in particular, concerning both mean and higher moments, will provide fruitful insights for macroeconomic modeling assumptions. 
#
# Lastly, the paper is indirectly related to the research that advocated for eliciting probabilistic questions measuring subjective uncertainty in economic surveys (<cite data-cite="manski_measuring_2004">(Manski, 2004)</cite>, <cite data-cite="delavande2011measuring">(Delavande et al. 2011)</cite>, <cite data-cite="manski_survey_2018">(Manski, 2018)</cite>). Although the initial suspicion concerning to people’s ability in understanding, using and answering probabilistic questions is understandable, <cite data-cite="bertrand_people_2001">(Bertrand and Mullainathan,2001)</cite> and other works have shown respondents have the consistent ability and willingness to assign a probability (or “percent chance”) to future events. <cite data-cite="armantier_overview_2017">(Armantier et al. 2017)</cite>  have a thorough discussion on designing, experimenting and implementing the consumer expectation surveys to ensure the quality of the responses. Broadly speaking, the advocates have argued that going beyond the revealed preference approach, availability to survey data provides economists with direct information on agents’ expectations and helps avoids imposing arbitrary assumptions. This insight holds for not only point forecast but also and even more importantly, for uncertainty, because for any economic decision made by a risk-averse agent, not only the expectation but also the perceived risks matter a great deal.
#

# + {"code_folding": [0], "hide_output": true}
## import libraries for inserting figures 

import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline 
from IPython.display import display, Image
import matplotlib.image as mpimg
import os
import pandas as pd

path = os.getcwd()

"""
def in_ipynb():
    try:
        if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
            return True
        else:
            return False
    except NameError:
        return False

# Determine whether to make the figures inline (for spyder or jupyter)
# vs whatever is the automatic setting that will apply if run from the terminal
if in_ipynb():
    # %matplotlib inline generates a syntax error when run from the shell
    # so do this instead
    get_ipython().run_line_magic('matplotlib', 'inline') 
else:
    get_ipython().run_line_magic('matplotlib', 'auto') 
    
"""
# -

# # Data, variables and density estimation
#
# ## Data
#
# The data used for this paper is from the core module of Survey of Consumer Expectation(SCE) conducted by the New York Fed, a monthly online survey for a rotating panel of around 1,300 household heads. The sample period in my paper spans from June 2013 to June 2019, in a total of 72 months. This makes about 85852 household-year observations, among which around 55,481 observations provide non-empty answers to the density question on earning growth. 
#
# Particularly relevant for my purpose, the questionnaire asks each respondent to fill perceived probabilities of their same-job-hour earning growth to pre-defined non-overlapping bins. The question is framed as "suppose that 12 months from now, you are working in the exact same [“main” if Q11>1] job at the same place you currently work and working the exact same number of hours. In your view, what would you say is the percentage chance that 12 months from now: increased by x% or more?".
#
# As a special feature of the online questionnaire, the survey only moves on to the next question if the probabilities filled in all bins add up to one. This ensures the basic probabilistic consistency of the answers crucial for any further analysis. Besides, the earning growth expectation regarding exactly the same position, same hours, and the same location has two important implications for my analysis. First, these requirements help make sure the comparability of the answers across time and also excludes the potential changes in earnings driven by endogenous labor supply decisions, i.e. working for longer hours. Second, the earning expectations and risks measured here are only conditional on non-separation from the current job. It excludes either unemployment, i.e. likely a zero earning, or an upward movement in the job ladder, i.e. a different earning growth rate. Therefore, this does fully reflect the entire income risk profile relevant to each individual. 
#
# Unemployment and other involuntary job separations are undoubtedly important sources of income risks, but I choose to focus on the same-job/hour earning with the recognition that individuals' income expectations, if any, may be easier to be formed for the current job/hour than when taking into account unemployment risks. Given the focus of this paper being subjective perceptions, this serves as a  useful benchmark.  What is more assuring is that the bias due to omission of unemployment risk is unambiguous. We could interpret the moments of same-job-hour earning growth as an upper bound for the level of growth rate and a lower bound for the income risk. To put it in another way, the expected earning growth conditional on current employment is higher than the unconditional one, and the conditional earning risk is lower than the unconditional one. At the same time, since SCE separately elicits the perceived probability of losing the current job for each respondent, I could adjust the measured labor income moments taking into account the unemployment risk. 
#
# ## Density estimation and variables 
#
# With the histogram answers for each individual in hand, I follow <cite data-cite="engelberg_comparing_2009">(Engelberg, Manskiw and Williams, 2009)</cite> to fit each of them with a parametric distribution accordingly for three following cases. In the first case when there are three or more intervals filled with positive probabilities, it will be fitted with a generalized beta distribution. In particular, if there is no open-ended bin on the left or right, then two-parameter beta distribution is sufficient. If there is either open-ended bin with positive probability, since the lower bound or upper bound of the support needs to be determined, a four-parameter beta distribution is estimated. In the second case, in which there are exactly two adjacent intervals with positive probabilities, it is fitted with an isosceles triangular distribution. In the third case, if there is only one positive-probability of interval only, i.e. equal to one, it is fitted with a uniform distribution. 
#
# Since subjective moments such as variance is calculated based on the estimated distribution, it is important to make sure the estimation assumptions of the density distribution do not mechanically distort my cross-sectional patterns of the estimated moments. This is the most obviously seen in the tail risk measure, skewness. The assumption of log normality of income process, common in the literature (See again <cite data-cite="blundell_consumption_2008">(Blundell et al. 2008)</cite>), implicitly assume zero skewness, i.e. that the income increase and decrease from its mean are equally likely. This may not be the case in our surveyed density for many individuals. In order to account for this possibility, the assumed density distribution should be flexible enough to allow for different shapes of subjective distribution. Beta distribution fits this purpose well. Of course, in the case of uniform and isosceles triangular distribution, the skewness is zero by default. 
#
# Since the microdata provided in the SCE website already includes the estimated mean, variance and IQR by the staff economists following the exact same approach, I directly use their estimates for these moments. At the same time, for the measure of tail-risk, i.e. skewness, as not provided, I use my own estimates. I also confirm that my estimates and theirs for the first two moments are correlated with a coefficient of 0.9. 
#
# For all the moment's estimates, there are inevitably extreme values. This could be due to the idiosyncratic answers provided by the original respondent, or some non-convergence of the numeric estimation program. Therefore, for each moment of the analysis, I exclude top and bottom $3\%$ observations, leading to a sample size of around 48,000. 
#
# I also recognize what is really relevant to many economic decisions such as consumption is real income instead of nominal income. I, therefore, use the inflation expectation and inflation uncertainty (also estimated from density question) to convert nominal earning growth moments to real terms for some robustness checks in this paper. In particular, the real earning growth rate is expected nominal growth minus inflation expectation. 
#
#
# \begin{eqnarray}
# \overline {\Delta y^{r}}_{i,t} = \overline\Delta y_{i,t} - \overline \pi_{i,t}
# \end{eqnarray}
#
# The variance associated with real earning growth, if we treat inflation and nominal earning growth as two independent stochastic variables, is equal to the summed variance of the two. The independence assumption is admittedly an imperfect assumption because of the correlation of wage growth and inflation at the macro level. So it is should be interpreted with caution. 
#
# \begin{eqnarray}
# \overline{var^{r}}_{i,t} = \overline{var}_{i,t} + \overline{var}_{i,t}(\pi_{t})
# \end{eqnarray}
#
# Not enough information is available for the same kind of transformation of IQR and skewness from nominal to real, so I only use nominal variables. Besides, as there are extreme values on inflation expectations and uncertainty, I also exclude top and bottom $5\%$ of the observations. This further shrinks the sample, when using real moments, to 36,000. 
#

# #  Perceived income risks: basic facts 
#
#
# ##  Cross-sectional heterogeneity
#
# This section inspects some basic cross-sectional patterns of the subject moments of labor income. In the Figure \ref{fig:histmoms} below, I plot the cross-sectional distribution of $\overline\Delta y_{i,t}$, $\overline{var}_{i,t}$, $\overline {\Delta y^{r}}_{i,t}$, $\overline{var^{r}}_{i,t}$. 
#
# First, expected income growth across the population exhibits a dispersion ranging from a decrease of $2-3\%$ to around an increase of $10\%$ in nominal terms. Given the well-known downward wage rigidity, it is not surprising that most of the people expect a positive earning growth. At the same time, the distribution of expected income growth is right-skewed, meaning that more workers expect a smaller than larger wage growth. What is interesting is that this cross-sectional right-skewness in nominal earning disappears in expected real terms. Expected earnings growth adjusted by individual inflation expectation becomes symmetric around zero, ranging from a decrease of $10\%$ to an increase of $10\%$. Real labor income increase and decrease are approximately equally likely.  
#
# Second, as the primary focus of this paper, perceived income risks also have a notable cross-sectional dispersion. In terms of both nominal and real terms, the distribution is right-skewed with a long tail. Specifically, most of the workers have perceived a variance of nominal earning growth ranging from zero to $20$ (a standard-deviation equivalence of $4-4.5\%$ income growth a year). But in the tail, some of the workers perceive risks to be as high as $7-8\%$ standard deviation a year. To have a better sense of how large the risk is, consider a median individual in our sample, who has an expected earning growth of $2.4\%$, and a perceived risk of $1\%$ standard deviation. This implies by no means negligible earning risk. Lastly, in the figures not presented here, it is found about half of the sample exhibits non-zero skewness in their subjective distribution, an indicator of symmetry of the perceived density or upper/lower tail risk. 
#
# <center>[FIGURE \ref{fig:histmoms} HERE]</center>

# + {"caption": "Distribution of Individual Moments", "code_folding": [], "label": "fig:histmoms", "note": "this figure plots histograms of the individual income moments. inc for nominal and rinc for real.", "hide_input": true, "hide_output": true}
## insert figures

graph_path = os.path.join(path,'../Graphs/ind/')

fig_list = ['hist_incexp.jpg',
            'hist_rincexp.jpg',
            'hist_incvar.jpg',
            'hist_rincvar.jpg']
            
nb_fig = len(fig_list)
    
file_list = [graph_path+ fig for fig in fig_list]

## show figures 
plt.figure(figsize=(18,10))
for i in range(nb_fig):
    plt.subplot(int(nb_fig/2),2,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")
# -

# ##  Counter-cylicality 
#
# This section shows that the perceived income risks are countercylical, i.e. perceived risks are negatively correlated with the recent labor market outcomes. 
#
#
# <center>[FIGURE \ref{fig:tshe} HERE]</center>
#
#
# <center>[TABLE \ref{macro_corr_he} HERE]</center>
#
# <center>[TABLE \ref{macro_corr_he_state} HERE]</center>

# + {"caption": "Recent Labor Market Outcome and Perceived Risks", "code_folding": [], "label": "fig:tshe", "note": "Recent labor market outcome is measured by hourly earning growth (YoY).", "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/pop/')

fig_list = ['tsMeanexp_he.jpg',
            'tsMeanvar_he.jpg']
            
nb_fig = len(fig_list)

file_list = [graph_path+fig for fig in fig_list]


## show figures 

fig, ax = plt.subplots(figsize =(90,30),
                       nrows = nb_fig,
                       ncols = 1)
for i in range(nb_fig):
    ax[i].imshow(mpimg.imread(file_list[i]))
    ax[i].axis("off") 
plt.tight_layout(pad =0, w_pad=0, h_pad=0)

# + {"hide_input": true, "hide_output": true}
macro_corr  = pd.read_excel('../Tables/macro_corr_he.xlsx',index_col=0)
print('Correlation of perceived risks and past labor market conditions')
macro_corr

# + {"hide_input": true, "hide_output": true}
mom_group_state  = pd.read_excel('../Tables/mom_group_state.xls',index_col = 0)
print('Perceived income risk and state labor market condition')
mom_group_state = mom_group_state.replace(np.nan, '', regex=True)
mom_group_state
# -

# ## Experienced volatility and perceived risks 
#
# This section shows that experienced volatility shapes perceived income risks. 
#
# <center>[FIGURE \ref{fig:var_experience_var_data} HERE]</center>

# + {"caption": "Experienced Volatility and Perceived Income Risk", "code_folding": [], "label": "fig:var_experience_var_data", "note": "Experienced volatility is the mean squred error(MSE) of income regression based on a particular year-cohort sample. The perceived income risk is the average across all individuals from the cohort in that year.", "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/ind/')

fig_list = ['experience_var_var_data.png']
            
nb_fig = len(fig_list)

file_list = [graph_path+fig for fig in fig_list]



## show figures 
plt.figure(figsize =(20,20))
for i in range(nb_fig):
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off") 
plt.tight_layout(pad =0, w_pad=0, h_pad=0)

# + {"hide_input": true, "hide_output": true}
micro_reg_history_vol  = pd.read_excel('../Tables/micro_reg_history_vol.xls',index_col = 0)
print('Experienced volatility and perceived income risk')
micro_reg_history_vol = micro_reg_history_vol.replace(np.nan, '', regex=True)
micro_reg_history_vol
# -

#
#
# ##  Role of individual characteristics
#   
#    What factors are associated with subjective riskiness of labor income? This section inspects the question by regressing the perceived income risks at individual level on three major blocks of variables: job-specific characteristics, household demographics and other macroeconomic expectations held by the respondent. 
#
# In a general form, the regression is specified as followed, where the dependent variable is one of the individual subjective moments that represent perceived income risks for either nominal or real earning. 
#
# \begin{eqnarray}
# \{\overline{var}_{i,t}, \overline{var}^r_{i,t}, \overline{iqr}_{i,t}\} = \alpha + \beta_0 \textrm{HH}_{i,t} + \beta_1 \textrm{JobType}_{i,t} + \beta_2 \textrm{Exp}_{i,t} + \beta_3 \textrm{Month}_t + \epsilon_{i,t}
# \end{eqnarray}
#
# The first block of factors, as called $\textit{Jobtype}_{i,t}$ includes dummy variables indicating if the job is part-time or if the work is for others or self-employed. Since the earning growth is specifically asked regarding the current job of the individual, I can directly test if a part-time job and the self-employed job is associated with higher perceived risks. 
#
# The second type of factors denoted $\textit{HH}_{i,t}$ represents household-specific demographics such as the household income level, education, and gender of the respondent. 
#
# Third, $\textit{Exp}_{i,t}$ represents other subjective expectations held by the same individual. As far as this paper is concerned, I include the perceived probability of unemployment herself, the probability of stock market rise over the next year and the probability of a higher nationwide unemployment rate. 
#
# $\textit{Month}_t$ is meant to control possible seasonal or month-of-the-year fixed effects. It may well be the case that at a certain point of the time of the year, workers are more likely to learn about news to their future earnings. But as I have shown in the previous section, such evidence is limited particularly for the higher moments of earnings growth expectations. 
#
# Besides, since many of the regressors are time-invariant household characteristics, I choose not to control household fixed effects in these regressions ($\omega_i$). Throughout all specifications, I cluster standard errors at the household level because of the concern of unobservable household heterogeneity. The regression results are presented in Table \ref{micro_reg} below for three measures of perceived income risks, nominal growth variance, nominal growth IQR, and real growth variance. 
#
# The regression results are rather intuitive. It confirms that self-employed jobs, workers from low-income households and lower education have higher perceived income risks. In our sample, there are around $15\%$ (6000) of the individuals who report themselves to be self-employed instead of working for others. In the Table \ref{micro_reg_mean} shown in the appendix, this group of people also has higher expected earnings growth. The effects are statistically and economically significant. Whether a part-time job is associated with higher perceived risk is ambiguous depending on if we control household demographics. At first sight, part-time jobs may be thought of as more unstable. But the exact nature of part-time job varies across different types and populations. It is possible, for instance, that the part-time jobs available to high-income and educated workers bear lower risks than those by the low-income and low-education groups. 
#
# The negative correlation between perceived risks and household income is significant and robust throughout all specifications. In contrast, there is no such correlation between expected earning growth per se and household income. Although SCE asks the respondent to report an income range instead of the accurate monetary value, the 11-group breakdown is sufficiently granular to examine if the high-income/low risks association is monotonic. As implied by the size of the coefficient of each income group dummy in the Table \ref{micro_reg}, this pattern is monotonically negative until the top income group ($200k or above). I also plot the mean and median of income risks by income group in the Figure \ref{fig:boxplotbygroup}.  
#
# Besides household income, there is also a statistical correlation between perceived risks and other demographic variables. In particular, higher education, being a male versus female, being a middle-aged worker compared to a young, are all associated with lower perceived income risks. To keep a sufficiently large sample size, I run regressions of this set of variables without controlling the rest regressors.  Although the sample size shrinks substantially by including these demographics, the relationships are statistically significant and consistent across all measures of earning risks. 
#
# Higher perceived the probability of losing the current job, which I call individual unemployment risk, $\textit{IndUE}$ is associated with higher earning risks of the current job. The perceived chance that the nationwide unemployment rate going up next year, which I call aggregate unemployment risk, $\textit{AggUE}$ has a similar correlation with perceived earning risks. Such a positive correlation is important because this implies that a more comprehensively measured income risk facing the individual that incorporates not only the current job's earning risks but also the risk of unemployment is actually higher. Moreover, the perceived risk is higher for those whose perceptions of the earning risk and unemployment risk are more correlated than those less correlated. 
#
# Lastly, what is ambiguous from the regression is the correlation between stock market expectations and perceived income risks. Although a more positive stock market expectation is associated with higher expected earnings growth in both real and nominal terms, it is positively correlated with nominal earning risks but negatively correlated with real earning risks. As the real earning risk is the summation of the perceived risk of nominal earning and inflation uncertainty, the sign difference has to be driven by a negative correlation of expectation stock market and inflation uncertainty.  In order to reach more conclusive statements, I will examine how perceived labor income risks correlate with the realized stock market returns and indicators of business cycles depending upon individual factors in the next step of the analysis. 
#
# To summarize, a few questions arise from the patterns discussed above. First, what drives the differences in subjective earning risks across different workers? To what extent these perceptive differences reflect the true heterogeneity of the income risks facing by these individuals? Or they can be attributed to perceptive heterogeneity independent from the true risk profile. Second, how are individual earning risk is correlated with asset return expectations and broadly the macroeconomic environment? This will be the focus of the coming sections. 
#      
#     
#  <center>[FIGURE \ref{fig:barplot_byinc} HERE]</center>
#  
#  <center>[FIGURE \ref{fig:ts_incvar_age} HERE]</center>
#  
#

# + {"caption": "Perceived Income by Income", "code_folding": [], "label": "fig:barplot_byinc", "note": "this figure plots average perceived income risks by the range of household income.", "widefigure": true, "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/ind/')

fig_list = ['boxplot_var_stata.png']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(20,20))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")

# + {"caption": "Perceived Income by Age", "code_folding": [], "label": "fig:ts_incvar_age", "note": "this figure plots average perceived income risks of different age groups over time.", "widefigure": true, "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/ind/ts/')

fig_list = ['ts_incvar_age_g_mean.png']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(15,15))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")
# -

reg_tb = pd.read_excel('../Tables/micro_reg.xlsx').replace(np.nan,'')

# + {"hide_input": true, "hide_output": true}
reg_tb
# -

# ##  Perceived income risks and decisions
#
# This section investigates how individual-specific perceived risks are correlated with household economic decisions such as consumption and labor supply. I should note that the purpose of this exercise is not primarily for causal inference at the current stage. Instead, it is meant to check if the surveyed households demonstrate a certain degree of in-survey consistency in terms of their perceptions and decision inclinations. 
#
# In particular, I ask two questions based on the available survey answers provided by the core module of the survey. First, are higher perceived income risks associated with a lower anticipated household spending growth? Second, are higher perceived income risks are associated with actions of self-insurance such as seeking an alternative job. This can be indirectly tested using the surveyed probability of voluntary separation from the current job. In addition, supplementary modules of SCE have also surveyed more detailed questions on spending decisions and the labor market. (These I will examine in the next stage of the analysis.)
#
# There is one important econometric concern when I run regressions of the decision variable on perceived risks due to the measurement error in the regressor used here. In a typical OLS regression in which the regressor has i.i.d. measurement errors, the coefficient estimate for the imperfectly measured regressor will have a bias toward zero. For this reason, if I find that willingness to consume is indeed negatively correlated with perceived risks, taking into account the bias, it implies that the correlation of the two is greater in the magnitude. 
#
#
# [TABLE \ref{spending_reg} HERE]

spending_reg_tb = pd.read_excel('../Tables/spending_reg.xlsx').replace(np.nan,'')

# + {"hide_input": true, "hide_output": true}
spending_reg_tb
# -

# #  Model 
#
# ## Income process and a model of learning  
#
# We start by defining an AR(1) process of the individual income. In particular, the income of individual $i$ from the cohort $c$ at time $t$ depends on her previous-period income with a persistence parameter of $\rho$ and an individual and time-specific shock $\epsilon_{i,c,t}$. I define cohort $c$ to be measured by the year of entry in the job market.
#
# \begin{eqnarray}
# y_{i,c,t} = \rho y_{i,c,t-1} + \epsilon_{i,c,t}
# \end{eqnarray}
#
# It is assumed that the $\rho$ is the same across all inviduals. Also, I assume the income shock $\epsilon_{i,c,t}$ to be i.i.d., namely independent across individuals and the time,and with an identical variance, as defined in the equation below. Later sections will relax this assumption by allowing for cross-sectional correlation, namely some aggregate risks. Further extensions are also allowed for cohort or time-specific volatility. The i.i.d. assumption implies at any time $t$, variance-covariance matrix of income shocks across individuals have is a diagonal matrix.
#
# \begin{eqnarray}
# E(\epsilon_{t}'\epsilon_{t}|Y_{t-1}) = \sigma^2 I_n \quad \forall t 
# \end{eqnarray}
#
# where $\sigma^2$ is the volatility of income shock and $I_n$ is an identity matrix whose length is the number of agents in the economy, $n$. Although income volatility is not cohort-specific, any past shock still created different impacts on the young and old generations because their length of the proefessional career are different. This is reminiscent of <cite data-cite="bansal2004risks">Storesletten et al. (2004)</cite>. Since both $\rho$ and $\sigma^2$ are not cohort-specific, I drop the subscript $c$ from now on to avoid clustering. 
#
# Both $\rho$ and $\sigma$ are "true" parameters only known by the modeler, but unknown by agents in the economy. Individual $i$ learns about the income process by "running" a regression based on the model above using a limited sample from her past experience starting from the year of entering the job market $c$ up till $t$. Critically, for this paper's purpose,  I allow the experience used for learning to include both her own and others' past income over the same period. It is admittedly bizarre to assume individual agents have access to the whole population's income. A more realistic assumption could be that only a small cross-sectional sample is available to the agent. Any scope of cross-sectional social learning suffices for the point to be made in this paper.  
#

#
# ### A baseline model of experience-based learning
#
# If each agent knows _perfectly_ the model parameters $\rho$ and $\sigma$, the uncertainty about future income growth is 
#
# \begin{eqnarray}
# \begin{split}
# Var^*_{i,t}(\Delta y_{i,t+1}) & =  Var^*_{i,t}(y_{i,t+1}- y_{i,t}) \\ 
# & =  Var^*_{i,t}((\rho-1)y_{i,t} + \epsilon_{i,t+1}) \\
# & = Var^*_{i,t}(\epsilon_{i,t+1}) \\
# & = \sigma^2
# \end{split}
# \end{eqnarray}
#
# The superscript $*$ is the notation for perfect understanding. The first equality follows because both $y_{i,t}$ and the persistent parameter $\rho$ is known by the agent. The second follows because $\sigma^2$ is also known. 
#
# Under _imperfect_ understanding and learning, both $\rho$ and $\sigma^2$ are unknown to agents. Therefore, the agent needs to learn about the parameters from the small panel sample experienced up to that point of the time. We represent the sample estimates of $\rho$ and $\sigma^2$ using $\widehat \rho$ and $\hat{\sigma}^2$. 
#
# \begin{eqnarray}
# \begin{split}
# \widehat Var_{i,t}(\Delta y_{i,t+1}) & = y_{i,t}^2 \underbrace{\widehat{Var}^{\rho}_{i,t}}_{\text{Persistence uncertainty}} + \underbrace{\hat{\sigma}^2_{i,t}}_{\text{Shock uncertainty}}
# \end{split}
# \end{eqnarray}
#
# The perceived risks of future income growth have two components. The first one comes from the uncertainty about the persistence parameter. It reflects how uncertain the agent feels about the degree to which realized income shocks will affect her future income, which is non-existent under perfect understanding. I will refer to this as the parameter uncertainty or persistence uncertainty hereafter. Notice the persistence uncertainty is scaled by the squared size of the contemporary income. It implies that the income risks are size-dependent under imperfect understanding. It introduces one of the possible channels via which current income affects perceived income risk. 
#
# The second component of perceived risk has to do with the unrealized shock itself. Therefore, it can be called shock uncertainty. Because the agent does not know perfectly the underlying volatility of the income shock, she makes an estimate based on past volatility. The estimates $\hat{\sigma}^2_{i,t}$ can be lower or higher than the true risks, but it is an unbiased estimator for the true $\sigma^2$. 
#
# We assume agents learn about the parameters using a least-square rule widely used in the learning literature (For instance, <cite data-cite="evans2012learning">Evans and Honkapohja (2012)</cite>, <cite data-cite="malmendier2015learning">Malmendier and Negal (2015)</cite>) The bounded rationality prevents her from adopting any more sophisticated rule that econometricians may consider to be superior to the OLS. (For instance, OLS applied in autocorrelated models induce bias in estimate.) We first consider the case when the agent understands that the income shocks are i.i.d. To put it differently, this is when the agent correctly specify the income model when learning. The least-square estimate of paramters are the following.
#
#
# \begin{eqnarray}
# \hat \rho_{i,t} = (\sum^{t-c}_{k=0}\sum^{n}_{j=1}y^2_{j,t-k-1})^{-1}(\sum^{t-c}_{k=0}\sum^{n}_{j=1}y_{j,t-k-1}y_{j,t-k})
# \end{eqnarray}
#
# The variance of sample residuls $\widehat e$ are used for estimating the income volatility $\sigma^2$. It can be seen as the experienced volatility over the past history. 
#
# \begin{eqnarray}
# \widehat{\sigma}^2_{i,t} = s^2_{i,t} = \frac{1}{N_{i,t}-1} \sum^{n}_{j=1}\sum^{t-c}_{k=0} \hat e_{j,t-k}^2
# \end{eqnarray}
#
# where $N_{i,t}$ is the size of the panel sample available to the agent $i$ at time t. It is equal to $n(t-c)$, the number of people in the sample times the duration of agent $i$'s career. 
#
# Under i.i.d. assumption, the estimated uncertainty about the estimate is 
#
# \begin{eqnarray}
# \widehat {Var}^{\rho}_{i,t} = (\sum^{t-c}_{k=0}\sum^{n}_{j=1}y^2_{j,t-k-1})^{-1}\widehat{\sigma}^2_{i,t}
# \end{eqnarray}
#
# Experience-based learning naturally introduces a mechanism for the perceived income risks to be cohort-specific and age-specific. Different generations who have experienced different realizations of the income shocks have different estimates of $Var^{\rho}$ and $\sigma^2$, thus differ in their uncertainty about future income. In the meantime, people at an older age are faced with a larger sample size than younger ones, this will drive the age profile of perceived risks in line with the observation that the perceived risk is lower as one grows older. Also, note that the learning literature has explored a wide variety of assumptions on the gains from learning to decline over time or age. These features can be easily incorporated into my framework. For now, equal weighting of the past experience suffices for the exposition here. 
#
# We can rewrite the perceived risk under correct model specification as the following. 
#
#
# \begin{eqnarray}
# \widehat{Var}_{i,t}(\Delta y_{i,t+1}) = [(\sum^{t-c}_{k=0}\sum^{n}_{j=1}y^2_{j,t-k-1})^{-1}y^2_{i,t} + 1] \hat{\sigma}^2_{i,t}
# \end{eqnarray}
#
#

#
# ### Attribution  
#
# Attribution means that agents subjectively form perceptions about the correlation between their own income outcome and others. This opens room for possible model-misspecification about the nature of income shock due to bounded rationality. Although people specify the regression model correctly, they do not necessarily perceive the nature of the income shocks correctly. 
#
# Before introducing the specific mechanism of the attribution error,  we can generally discuss the property of parameter uncertainty for any general subjective perception of the cross-sectional correlation. Under the least-square learning rule, the perceived uncertainty about the parameter estimate now takes a more general form as below. It is equivalent to accounting for within-time clustering in computing standard errors.
#
# \begin{eqnarray}
# \begin{split}
# \tilde {Var}^{\rho}_{i,t} & =   (\sum^{t-c}_{k=0}\sum^{n}_{j=1}y^2_{j,t-k-1})^{-1}(\sum^{t-c}_{k=0}\tilde \Omega_{i,t-k})(\sum^{t-c}_{k=0}\sum^{n}_{j=1}y^2_{j,t-k-1})^{-1}
# \end{split}
# \end{eqnarray}
#
# where $\tilde \Omega_{t-k}$ is the perceived variance-covariance of income and income shocks within each point of time. It reflect how individual $i$ thinks about the correlation between her own income and others'. 
#
# \begin{eqnarray}
# \begin{split}
# \tilde \Omega_{i,t} = \tilde E_{i,t}(Y_{t-1}e_{t}'e_{t}Y_{t-1})
# \end{split}
# \end{eqnarray}
#
# If we assume constant group size $n$ over time and the homoscedasticity, i.e. income risks $\sigma$ do not change over time, given the individual ascribes a subjective correlation coefficient of $\tilde \delta_{\epsilon, i,t}$ across income shocks and a correlation $\tilde \delta_{y, i,t}$ across the levels of income, $\tilde \Omega_{i,t}$ can be approximated as the following. (See the appendix for derivation) (This is analogous to the cluster-robust standard error by <cite data-cite="cameron2011robust">Cameron et al. (2011)</cite>. But the distinction is that both long-run and short-run correlation are subjective now. ) 
#
#
# \begin{eqnarray}
# \begin{split}
# \tilde \Omega_{t} & \approx \sum^{n}_{j=1}y^2_{j,t} (1+\tilde \delta_{y,i,t}\tilde \delta_{\epsilon,i,t}(n-1))\tilde \sigma^2_{t}
# \end{split}
# \end{eqnarray}
#
# Therefore, the parameter uncertainty under the subjective attribution takes a following form comparable with that derive for i.i.d. in previous section. 
#
# \begin{eqnarray}
# \begin{split}
# \tilde {Var}^{\rho}_{i,t} & = (\sum^{t-c}_{k=0}\sum^{n}_{j=1}y^2_{j,t-k-1})^{-1}(1+ \tilde\delta_{i,t}(n-1))\tilde{\sigma}^2_{t}
# \end{split}
# \end{eqnarray}
#
# Where we bundle the two correlation coefficients parameters together as a single parameter of the attribution correlation, which represents the degree of attribution errors. 
#
# \begin{eqnarray}
# \tilde \delta_{y,i,t}\tilde \delta_{\epsilon,i,t}\equiv \tilde \delta_{i,t}  
# \end{eqnarray}
#
# The subjective attribution is jointly  by two perceived correlation parameters, $\tilde \delta_{\epsilon}$ and $\tilde \delta_y$. They can be more intuively thought as long-run attribution and short-run attribution, respectively, because the former is the perceived correlation in the level of the income and later in income shocks. The multiplication of two jointly governs the degree to which the agents inflate experienced volatility in forming perceptions about future income risks. 
# $\tilde \delta_{i,t} = 0$ if the agent $i$ thinks that her income shock or the long-run income is uncorrelated with others' ($\tilde \delta_{\epsilon} = 0$ or $\tilde \delta_y = 0$). In contrats, $\tilde \delta_{i,t} = 1$, attaining its maximum value if the agent thinks both her income shock and income is perfectly correlated with others. In general,  $\tilde \delta_{\epsilon,i,t}$ and $\tilde \delta_{y,i,t}$ are not necessarily consistent with the true income process. Since long-run correlation increases with the the short-run correlation, bundling them together as a single parameter is feasible. 
#
# Another important aspect regarding attribution is that it changes perceived risk only through its effect on parameter uncertainty but not on shock uncertainty. Attributing the individual outcome either to idiosyncrasy or common factors do not change how agents think of the variance of the shock, but changes the uncertainty about how persistent the effect of the shock will be. Therefore, for the attribution to play a meaningful role in perceived risk, the size of the income shock shall not be excessively so big that it overshadows the role of persistence uncertainty. 
#

# + {"caption": "Attribution and Parameter Uncertainty", "code_folding": [], "label": "fig:corr_var", "widefigure": true, "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/theory/')

fig_list = ['corr_var.jpg']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(10,10))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")
# -

# ### Attribution errors 
#
# The framework set up above can neatly incorporate the psychological tendency of ascribing bad luck to external causes and good luck to internal ones. The manifesto of the attribution error in this context is that people asymmetrically assign the subjective correlation $\tilde \delta_{i,t}$ depending on the sign of the recent income change (or the realized shocks). An internal attribution implies a positive change in income induces the agent to maintain the independence assumption, while an external attribution means a negative change in income makes the agent interpret the income shock as a common shock and thus positively correlated with others at each point of the time. More formally, we define the attribution error as the assymetric assignment of the value of $\tilde\delta_{i,t}$, specified as below. 
#
#
# \begin{eqnarray}
# \begin{split}
# \textrm{Internal attribution: }\quad \tilde\delta_{i,t} = 0 \quad \textrm{if} \quad \Delta y_{i,t}>0 \\
# \textrm{External attribution: }\quad \tilde\delta_{i,t} = 1 \quad \textrm{if} \quad \Delta y_{i,t}<0
# \end{split}
# \end{eqnarray}
#
# Here, I let the attribution be contingent on the income change $\Delta y_{i,t}$. An alternative way of modeling it is contingency on forecast errors, $\widehat e_{i,t}$, namely the unexpected income shock to agent $i$ at time $t$. The distinction between the two modeling techniques is indistinguishable in terms of qualitative predictions I will discuss next. 
#
#
# One can immediately show the following the persistence uncertainty with external attribution is no smaller than that with internal attribution. 
#
# \begin{eqnarray}
# \tilde {Var}^{\rho}_{i,t} \geq \widehat {Var}^{\rho}_{i,t} \quad \forall \quad \tilde\delta_{i,t} \geq 0
# \end{eqnarray}
#
# where the equality holds as a special case when $\tilde\delta_{i,t} = 0$. The left hand side monotonically increases with $\tilde \delta_{i,t}$. 
#
# In the meantime, the shock uncertainty estimate,$\sigma^2$ remain the same no matter if the attribution error arises, both of which are equal to the sample average of regression residuals $s^2$. 
#
#
# \begin{eqnarray}
# \tilde{\sigma}^2_{i,t} = \widehat{\sigma}^2_{i,t}
# \end{eqnarray}
#
# Combining the two relations above, one can show the perceived risks of an unlucky person is unambiguously higher than that of a lucky one. 
#
# \begin{eqnarray}
# \begin{split}
# \tilde {Var}_{i,t}(\Delta y_{i,t+1}) & = y_{i,t-1}^2 \tilde{Var}^{\rho}_{i,t} + \tilde{\sigma}^2_{i,t} \\
# & = [(\sum^{t-c}_{k=0}\sum^{n}_{j=1}y^2_{j,t-k-1})^{-1}(1+ \tilde\delta_{i,t}(n-1))y^2_{i,t} + 1] \tilde{\sigma}^2\\
# & \geq \widehat {Var}_{i,t}(\Delta y_{i,t+1}) 
# \end{split}
# \end{eqnarray}
#
# where, again, the equality holds without attribution errors, i.e. $\tilde \delta_{i,t} = 0$. One way to rephrase the inequality above is that the unlucky group excessively extrapolates the realized shocks into her perception of risks. There is no distinction between the two groups if there is no attribution errors. 
#
# We have the following predictions about the perceived income risks from the analysis. 
#
# - Higher experienced volatility, measured by $s^2 \equiv \tilde{\sigma}^2_{i,t}$ leads to higher perceived income risks. 
# - In the same time, future perceptions of the risks inflate the past volatility by proportionately depending on their subjective attribution. A higher degree of external attribution reflected by a higher $\tilde \delta_{i,t}$ implies a higher inflaiton of past volatility into future.  (See Figure \ref{fig:corr_var}.)
#
# - With attribution errors, people project past experienced volatility into perceived risks disproportionately depending on the subjective attribution. A higher perceived attribution to common shocks, a bigger $\tilde \delta_{i,t}$ induces a higher perceived risk. See the comparison between Figure \ref{fig:var_experience_var}. This is different from the scenario without attribution errors.
#
# It is important to note that this difference still arises even if one assumes the underlying shocks are indeed non-independent. Although different types of income shocks have different implications as to which group correctly or mis-specifies the model, it does not alter the distinction between the lucky and unlucky group. To put it bluntly,  the underlying process determines who is over-confident or under-confident. But the lucky group is always more confident than the unlucky group. 

# + {"caption": "Experienced Volatility and Perceived Risk", "code_folding": [], "label": "fig:var_experience_var", "widefigure": true, "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/theory/')

fig_list = ['var_experience_var.jpg']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(10,10))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")
# -

#
# ### Extrapolative attribution 
#
# The baseline model only lets the sign of the recent income change induce attribution errors, and assumes away the possibility of the attribution errors to depend on the magnitude of the recent changes endogenously. This is reflected in the model assumption that $\tilde \delta_i$ could take either 1 or 0 depending on the sign of the recent income change. We could alternatively allow the attributed correlation $\tilde \delta_i$ to be a function of the $\Delta(y_{i,t})$. This will open the room for income changes of different salience to induce different degrees of attribution errors. 
#
# In order to capture this size-dependent pattern, I choose an attribution function that takes the following form as the following. It does not have to be this function in particular, but its properties suit the purpose here.   
#
# \begin{eqnarray}
# \begin{split}
# \tilde \delta(\Delta y_{i,t}) = 1- \frac{1}{(1+e^{\alpha-\theta \Delta y_{i,t}})}
# \end{split}
# \end{eqnarray}
#
#
# Basically, the attribution function is a variant of a logistic function with its function value bounded between $[0,1]$. It takes an s-shape and the parameter $\theta$ governs the steepness of the s-shape around its input value. $\alpha$ is adjustable parameter chosen such that the attribution free of errors happen to be equal to the true correlation between individuals $\delta$, which here takes values of zero for independence assumption. In the model, $\theta$ is the parameter that governs the degree of the attribution errors. It takes any non-negative value. Although the qualitative pattern induced by the attribution errors stands for any positive $\theta$, letting it be a parameter leaves modelers the room to recover it from subjective risks data. The attribution function under different $\theta$ is shown in Figure \ref{fig:theta_corr}. The higher $\theta$ is, the more sensitive the assigned correlation is to the size of the shock, thus inducing a higher dispersion of the perceived correlation between the lucky group and the unlucky group. 

# + {"caption": "Attribution Function", "code_folding": [], "label": "fig:theta_corr", "widefigure": true, "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/theory/')

fig_list = ['theta_corr.jpg']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(10,10))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")

# + {"caption": "Current Income and Perceived Risk", "code_folding": [], "label": "fig:var_recent", "widefigure": true, "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/theory/')

fig_list = ['var_recent.jpg']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(10,10))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")
# -

# ## Simulation 
#
# ### Current income and perceived risks
#
# How do perceived risks depend on the current income level of $y_{i,t}$? Since the recent income changes $\Delta y_{i,t}$ triggers asymmetric attribution, the perceived risks depend on the current level of income beyond the past-dependence of future income on current income that is embodied in the AR(1) process. In particular, $\widehat{Var}^\rho_{i,t}$ does not depend on $\Delta y_{i,t}$ while $\tilde{Var}^\rho_{i,t}$ does and is always greater than the former as a positive, it will amplify the loading of the current level of income into perceived risks about future income. This generates a U-shaped perceived income profile depending on current level income.  
#
# Figure \ref{fig:var_recent} and \ref{fig:var_recent_sim} plots both the theory-predicted and simulated correlation between $y_{i,t}$ and perceived income risks with/without attribution errors. In the former scenario, perceived risks only mildly change with current income and the entire income profile of perceived risk is approximately flat. In the latter scenario, in contrast, perceived risks exhibit a clear U-shape across the income distribution. People sitting at both ends of the income distribution have high perceived risks than ones in the middle. The non-monotonic of the income profile arise due to the combined effects directly from $y_{i,t}$ and indirectly via its impact on $\tilde Var^{\rho}$. The former effect is symmetric around the long-run average of income (zero here). Deviations from the long-run mean on both sides lead to higher perceived risk. The latter monotonically decreases with current income because higher income level is associated with a more positive income change recently. The two effects combined create a U-shaped pattern.
#
# A subtle but interesting point is that the U-shape is skewed toward left, meaning perceived risks decrease with the income over the most part of the income distribution before the pattern reverses. More intuitively, it means that although low and high income perceived risks to be higher because of its deviation from the its long-run mean. This force is muted for the high income group because they have a lower peceived risks due to the attribution errors. 

# + {"caption": "Simulated Income Profile of Perceived Risk", "code_folding": [], "label": "fig:var_recent_sim", "widefigure": true, "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/theory/')

fig_list = ['var_recent_sim.jpg']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(10,10))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")
# -

# ### Age and experience and perceived risks 

# + {"caption": "Simulated Experience of Volatility and Perceived Risk", "code_folding": [], "label": "fig:var_experience_var_sim", "widefigure": true, "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/theory/')

fig_list = ['var_experience_var_sim.jpg']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(10,10))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")

# + {"caption": "Simulated Age Profile of Perceived Risk", "code_folding": [], "label": "fig:var_age_sim", "widefigure": true, "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/theory/')

fig_list = ['var_age_sim.jpg']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(10,10))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")
# -

#
# ### Aggregate risk
#
# Previously, I assume the underlying shock is i.i.d. This section considers the implication of the attribution errors in the presence of both aggregate and idiosyncratic risks. This can be modeled by assuming that the shocks to individuals' income are positively correlated with each other at each point of the time. Denoting $\delta>0$ as the true cross-sectional correlation of income shocks, the conditional variance-covariance of income shocks within each period is the following. 
#
#
# \begin{eqnarray}
# \begin{split}
# E(\epsilon_{t}'\epsilon_{t}|Y_{t-1}) = \Sigma^2 = \sigma^2\Omega \quad \forall t  
# \end{split}
# \end{eqnarray}
#
# where $\Omega$ takes one in its diagonal and $\delta$ in off-diagonal.  
#
# The learning process and the attribution errors all stay the same as before. Individuals specify their subjective structure of the shocks depending on the sign and size of their own experienced income changes. By the same mechanism elaborated above, a lucky person has lower perceived risks than her unlucky peer at any point of the time. This distinction between the two group stays the same even if the underlying income shocks are indeed correlated. 
#
# What's new in the presence of aggregate risks lies in the behaviors of average perceived risks, because there is an aggregate shock that drives the comovement of the income shocks affecting individuals. Compared to the environment with pure idiosyncratic risks, there is no longer an approximately equal fraction of lucky and unlucky agents at a given time. Instead, the relative fraction of each group depends on the recently realized aggregate shock. If the aggregate shock is positive, more people have experienced good luck and may, therefore, underestimate the correlation (a smaller $\tilde \delta$). This drives down the average perceived income risks among the population. If the aggregate shock is negative, more people have just experienced income decrease thus arriving at a higher perceived income uncertainty. 
#
# This naturally leads to a counter-cyclical pattern of the average perceived risks in the economy. The interplay of aggregate risks and attribution errors adds cyclical movements of the average perceived risks. The two conditions are both necessary to generate this pattern. Without the aggregate risk, both income shocks and perceived income shocks are purely idiosyncratic and they are averaged out in the aggregate level. Without attribution errors, agents symmetrically process experiences when forming future risk perceptions.
#
# Figure \ref{fig:recent_change_var_sim1} illustrates the first point. The scatter plots showcase the correlation between average income changes across population and average perceive risks under purely idiosyncratic risks and aggregate risks. The negative correlation with aggregate risks illustrate the counter-cylical perceived risks. There is no such a correlation under purely idiosyncratic risks. Figure \ref{fig:recent_change_var_sim2} testifies the second point. It plots the same correlation with and without attribution errors when the aggregate risk exists. Attribution errors brings about the asymmetry not seen when the bias is absent. 

# + {"caption": "Simulatd Average Labor Market and Perceived Risk", "code_folding": [], "label": "fig:recent_change_var_sim1", "widefigure": true, "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/theory/')

fig_list = ['var_recent_change_sim.jpg']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(70,45))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")

# + {"caption": "Simulated Average Labor Market Outcome and Perceived Risk", "code_folding": [], "label": "fig:recent_change_var_sim2", "widefigure": true, "hide_input": true, "hide_output": true}
## insert figures 
graph_path = os.path.join(path,'../Graphs/theory/')

fig_list = ['var_recent_change_sim2.jpg']
            
nb_fig = len(fig_list)
file_list = [graph_path+fig for fig in fig_list]

## show figures 
plt.figure(figsize=(70,45))
for i in range(nb_fig):
    plt.subplot(nb_fig,1,i+1)
    plt.imshow(mpimg.imread(file_list[i]))
    plt.axis("off")
# -

#
# #  Coclusion
#
# How do people form perceptions about their income risks? Theoretically, this paper builds an experience-based learning model that features an imperfect understanding of the size of the risks as well as its nature. By extending the learning-from-experience into a cross-sectional setting, I introduce a mechanism in which future risk perceptions are dependent upon past income volatility or cross-sectional distributions of the income shocks. I also introduce a novel channel - subjective attribution, into the learning to capture how income risk perceptions are also affected by the subjective determination of the nature of income risks.  It is shown that the model generates a few testable predictions about the relationship between experience/age/income and perceived income risks. 
#
# Empirically, I utilize recently available panel of income density surveys of U.S. earners to shed light directly on a subjective income risk profiles. I explore the cross-sectional heterogeneity in income risk perceptions across ages, generations, and income group as well as its cyclicality with the current labor market outcome. I found that risk perceptions are positively correlated with experienced income volatility, therefore differing across age and cohorts. I also found perceived income risks of earners counter-cyclically react to the recent labor market conditions. 
#
#
# As a next step, the project will build the experience-based-learning based on subjective attribution model into an otherwise life cycle model of consumption/saving. 
