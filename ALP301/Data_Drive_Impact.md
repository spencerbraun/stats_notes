[TOC]

# Data Driven Impact

* Learning objectives for technical members: learning the real world limitations and contexts for our models, seeing how learning methods are deployed in real companies, going beyond learning the methods from classes and understanding what is effective. How do we choose a learning method given limitations or existing data.
* Adaptive experiments, bandits, reinforcement learning the frontier replacing A/B testing. Adaptive experiments where the probability you receive certain emails can update every day or week. Creates faster learning

## Class Topics

### Machine Learning

* In slides, age is a confounder for death and treatment - correlated to both. Once included treatment effect increases and becomes signficant. 
* Treatment is for specific disease, so including disease is adding a highly correlated covariate. Interpreting the individual coefficients becomes more difficult, while prediction may not be harmed.
* Lasso renders model uninterpretable in the way a normal regression may be. Correlated regressors will have selection performed, Lasso is not telling you one covariate is more important than another. Selection is for regularized prediction.
* Lasso coefficients non monotonic in increasing lambda - shrinkage in a correlated predictor may allow a given coefficient to increase before decreasing as lambda increases
* Post lasso - use lasso for model selection, then run a regular regression on the selected predictors

### Causal Inference

* Causal interpretation is like missing data in machine learning. If it is missing at random, this is a relatively simple problem to solve. But if it is not missing at random and the data we observe has selection effects, then the imputation methods are not enough. 
* These are hard problems to solve and skepticism should be applied to our results.

## The Power of Experiments

### Chapter 4: Experimentation Considerations

* Barriers to experimentation
  * Not enough participants - achieving statistical power for certain kinds of randomization can be difficult - consider the level of randomization needed to understand if power can be achieved
  * Randomization can be hard to implement - communications are an easy touchpoint for experimentation, along with tech / digital infrastructure 
  * Experiments require data to measure their impact - choosing the right metric is challenging, the easiest to measure may skew incentives. Start with data audit to understand what data you already have, then consider additional data needed for desired inferences.
  * Underappreciation of decision makers’ unpredictability - people sometimes make weird and unpredictable decisions.
  * Overconfidence in our ability to guess the effect of an intervention 
* Log experiments centrally so teams have access to past experiments and don't have to start from scratch
* Build infrastructure so non-statisticians can run successful experiments, with a centralized expert team designing the platform and providing specialized assistance.

## Running Randomized Evaluations

### Chapter 3: Asking the Right Questions

* We need to have good descriptive information on the participants and the context of the program, know the objectives of the program, and have good process indicators that will tell us whether the program we are evaluating was well implemented.
* Start with strategic questions - what are the goals, make sure they are well-defined.
* Then turn to descriptive questions - what is the reality on the ground, what are the problems faced.
* Process questions - how well is the program being implemented? Are things actually running smoothly? If current program is not carried out correctly, may have higher impact correcting current policies than trying new ones.
* Impact questions - did it work?
* A needs assessment is a collection of qualitative and quantitative information meant to provide descriptive info on the context of the situation. The basis for a program design, helps us find weaknesses in current approaches, develop methodology for impact assessment. May be entirely sufficient if there is no problem, the problems are not prioritized by participants.
* Lit review may reveal we don't need to experiment - already have promising results we could implement.
* Business case assessment - how cost effective are the interventions, considering your context. In your best-case scenario, would your findings be a cost-effective solution that could be rolled out? Requires assumptions about costs and impact, scenario analysis is helpful.
* Impact evaluations can help answer questions like which alternative to pursue, which elements of a program matter most, can we work on multiple problems concurrently.
* Eventually need to choose which questions to address with randomized evaluation. Consider factors like the potential influence of the information yielded by evaluation. 
* When the question we want to answer is crosscutting, we must design our evaluation carefully so that we are isolating just the one factor we want to examine - orthogonalize.

## Business Data Science

### Chapter 5: Experiments

* AB trial, which is Silicon Valley’s label for a completely randomized design wherein the experiment subjects are randomly assigned to treatment status.
* Suppose that the treatment variable is di = 0 for users i in group A and di = 1 for those in group B. Then the ATE is the mean difference in response, averaged over all other influences on y in your distribution of users, for option B instead of A: $A T E=\mathbb{E}[y | d=1]-\mathbb{E}[y | d=0]$
* To have a causal interpretation, we require that *d* is *independent* of all other factors that could influence *y*. In an *AB* trial, this independence is achieved through randomization. With randomization, ATE is just the different in means between groups, $\widehat{\mathrm{ATE}}=\bar{y}_{1}-\bar{y}_{0}$
* Assuming independence,  $\operatorname{SE}\left(\bar{y}_{1}-\bar{y}_{0}\right)=\sqrt{\frac{1}{n_{0}} \widehat{\operatorname{var}}\left(y_{i} | d_{i}=0\right)+\frac{1}{n_{1}} \widehat{\operatorname{var}}\left(y_{i} | d_{i}=1\right)}$, and our ATE CI $\bar{y}_{1}-\bar{y}_{0} \pm 2 \operatorname{se}\left(\bar{y}_{1}-\bar{y}_{0}\right)$
* *Regression adjustment* is sometimes advocated as offering a reduced-variance of ATE. Take observed covariates $x_i$ for each individual. Fit a linear model for each treatment group $\mathbb{E}[y | \boldsymbol{x}, d]=\alpha_{d}+\boldsymbol{x}^{\prime} \boldsymbol{\beta}_{d}$. Given a pooled coviariate average across treatment groups $\bar{x}$ the regression adjusted ATE estimator is $\widehat{\mathrm{ATE}}=\alpha_{1}-\alpha_{0}+\bar{x}^{\prime}\left(\beta_{1}-\beta_{0}\right)$. Sometimes referred to as *post-stratification*.
* Adjusting for differences due to covariates can reduce the variance in  the ATE estimate because of random imbalances across groups. We tend to advise against these adjustments unless you have small sample sizes and know of a few factors that have a large influence on the  response, since this can introduce strange biases while the variance reduction is small.
* AA tests - compare "treatment" and control groups that receive the same treatment - any differences can highlight problems in experimental / randomization design since effect difference should be zero.
* The moment you lose perfect randomization, you are forced to start making modeling assumptions. We can control for observable biases in a regression, such as family size on probability of being an individual in the treatment group.
* The full interaction model is more robust to heterogeneous (covariate-dependent) treatment effects. When there is a covariate that has a large effect on y, there is good reason to suspect that it also moderates the effect of d on y.
* Want to estimate standard errors on the randomization level when there is dependence among the outcome for individuals - eg households instead of individuals. We could also use blocked bootstrap to estimate standard errors when there is dependence among individual outcomes (resampling at the household level).
* Clustered SEs are also a common way of adjusting for correlated errors - uses an extension of the Huber–White heteroskedastic consistent (HC) variance. Use vcovCL in AER package in R.
* *Blocked randomized design*: split the growing area into fields of relative homogeneity and apply each treatment level within subregions of each field
* Matched pairs, common in medical trials, where two similar patients are paired and each given a different treatment

##### Near Experimental Designs

* Diff-in-Diff: strong assumptions about treatment independence. $\mathbb{E}\left[y_{i t}\right]=\alpha+\beta_{d} d_{i}+\beta_{t} t+\gamma d_{i} t$ The treatment effect of interest is *γ*: the coefficient on the *interaction* between *di* and *t*. Treatment versus control group membership is encoded as *di* = 1 if DMA *i* is in the treatment group, *di* = 0 otherwise; thus, SEM is *on* for DMA *i* at time *t unless t* × *di* = 1. Instead of clustered SEs, we can control for fixed effects with a model $\mathbb{E}\left[y_{i t}\right]=\alpha_{i}+d_{i} \beta_{d}+t \beta_{t}+\gamma d_{i} t$
* Regression Discontinuity: treatment allocation is determined by a threshold on some “forcing  variable,” and subjects that are close to the threshold, on either side, are comparable for causal estimation purposes. You know the only confounding variable that you need to control for: the forcing variable. In a strict RD, the treatment is *fully determined* by the forcing variable so you just need to control for that variable. Continuity assumption: assume that if the threshold were to move slightly, subjects switching  treatment groups will behave similarly to those near them in their *new* treatment group. In an RD design, you learn about the treatment effect only at the threshold. Fit an ordinary least squares line in a window of ±*δ* on either side of the threshold (or weight least squares with weights for distance from threshold). It is possible to use similar tools for analysis when the treatment threshold is not strict but rather *fuzzy*
* IV: the *instrument z* affects the response *only* through its influence on the treatment. In addition, there are *unobserved* factors or errors, say *e*, that have influence over both the treatment and response. We refer to the policy variable as *endogenous* to the response. It is jointly determined with the response as a function of unobserved factors or errors - a regression will have omitted variable bias. TSLS uses the IV to estimate d, then use our predicted d as the variable of interest for the second regression. To control for other covariates, include them in first and second stage regressions. Steps: 1) first stage regression 2) use first stage to predict values of d 3) use predicted values in second stage. We can manually use `lm` twice and fit sandwich SEs, or use package AER with `ivreg` to automatically get robust SEs.

