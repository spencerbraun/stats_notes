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
* Subclassification - weighting cig deaths rates by the number of cig smokers per age group is the weighting in a population of cig smokers. Death rate for the population of cig smokers
  * Then weight the cig death rates by the number of pipe smokers per age group - this increases the death rate since the cigar smokers are older and cig death rates are higher for older populations
  * We could use whatever weightings we want, but the key is to weight cigar and cigarette smokers using the same age group weights. Weighting on the confounder in the same way between groups
* Propensity weighting is essentially an extension of this idea.
* ATT - what is the effect of the treatment on the group that actually receives the treatment. Say if an older population takes a certain drug but younger do not, we probably only care about the treatment effect on those who actually take the drug (older) than those that don't (younger). We look at ATE when we are looking at the effect across the population, but since treatment and control may not be balanced at all, we may just care about those receiving treatment.
* Athey papers on ML methods for determining confounders and assigning propensity scores instead of hand selection. 
* In the business setting, we are only building recommendations on observables - they are controlling for the information collected on users
* Fundamental problem of causal inference - the treatment effect is not observed for any individual, as seen through the potential outcomes model.
* Assumption: conditional on observables, treatment is randomly assigned. If there are unobserved factors that determine who receives treatment, this assumption is violated. For example voluntary enlistment in the military will have unobserved factors that determine whether someone joins, but the draft might allow us to assume that observables (your birthdate) determines your enlistment. We need to predict the counterfactual outcomes for treatment group if they had not been treated
* Outcome modeling - build a model of how x and y are related for control, then use to predict for treatment outcomes. But highly dependent on model accuracy and common support. Doubly robust methods to reweighting AND outcome modeling 
  * If treated observations have higher X's on average, we have confounding in our data. We can see this from the unequal distribution of X along X and Y axes.  X is affecting assignment and our outcomes. X would not be a confounder if the slope were flat (not affecting Y's) or were equally distributed across X axis (treatment uncorrelated to values of X).
  * For the reweighting step, we can use propensity weighting. 
  * Outcome modeling would run a regression - the consider for control what would their outcomes have been if their X values had been higher using the regression model. Give the control group a boost to match outcomes with treatment. This is quite dependent on a robust regression - we may be projecting a long distance along our model. 
  * Combining these methods is much more robust to their shortcomings. Can run a regression weighted by the inverse propensity scores.
* With propensity scores, looking to have a lot of overlap in densities. If we have very low propensity scores, need to cap amount of reweighting since you may simply be refitting to noise. In the COVID example, propensity weighting does not change the difference in means much, but the standard error increases. Also ran a regression to control for more covariates, the treatment effect was quite similar giving us more confidence in results. Feel good about the observed confounders but then left to question our unobserved confounders. Having the results line up this closely is unusual, then left to argue for whether your assumptions behind the model chosen are reasonable. 
* Analysis of beta blockers is a robustness check - no biological reason why this treatment should help, but if we found a treatment effect then might be an indication that the unobservables are a problem. 

### Experimental Design

* Experiments to change behavior like UK debt letter - consider how it might change the long term equilibrium. Could we shift to a new equilibrium or exhibit some mean reversion as the novelty of the new approach becomes habituated. Facebook increasing ads, may cause you to post less. A single person posting less has little effect on user engagement, but if you shift all posters to a new equilibrium there could be a macro issue.
* Relevance vs Position in Search: Observational data shows that first position links get clicked 25% of time, much more than lower positions. This was taken as evidence that link promotion to 1st spot increases click - but confounded by the relevance of the search - the results are supposed to be in relevance order to begin with. Running an experiment, experimentally assign the page position different from the relevance ranking. 
* Got a causal interpretation for value of positions. Oberservational effect of demotion far higher than the causal effect. Athey performed this experiment at Bing to quantify the amount the European commission should fine Google. But note they kept the experiment small since this provided a worse experience for users. Something to think about in all experiments - tradeoff between statistical power and loss of experience.
* Often trying to improve personalization in experiments. The ML is more off the shelf than it used to be, but the getting it into practice and having success metrics can often be difficult. Measurable KPIs in the short term but are related to long term impact.
* The evaluation metrics, what are you measuring. Balance between measuring what you care about and going slower, or measuring proxies but moving faster. Think about YouTube recommender for narrow subset of your interests, but leading to political extremism. 
* Stratification is key in business setting - not just for size but also consider seasonal stratification. 
* Can segregate users to receive a single treatment at a time. Or could choose to assume treatments are additive and can control for different effects through regression. With smaller dataset, might need to have overlapping experiments, but perhaps could segregate by type of experiments - eg 1 user interface experiment per user; could treat this as additive or just ignore the overlap since there should be no interaction. Control groups often get polluted as well - algorithms often optimized over all users, so their experience may change based on how treatments change the behavior of other groups. Facebook has no-ad long term control groups to see the engagement effects of seeing ads.
* Power analysis - can think of signal in the denominator and noise in the numerator $n \geq\left(\frac{z_{\alpha}+\Phi^{-1}(1-\beta)}{\theta_{0}} \cdot 2 \sigma\right)^{2}$. Estimated treatment effect - when cannot run a trial, may have to work from first principles estimating. May be able to reason x number of people will be unresponsive to treatment, y number have already adjusted behavior etc. Also can use ML predictive algorithms to estimate the size of an effect but often too optimistic - best to push towards conservatism and really consider whether the experiment will be worth it.
* Tradeoff between additional user engagement vs cannibalization - the flashier the intervention, the more it may detract from other valuable properties. 
* Goal tree to prioritize innovation - if trying to increase revenue, well that is dependent on factors built from the ground up. 
* Segmenting KPIs - can have individual segments go up but overall average go down. Watch out for all kinds of Simpson's paradoxes etc.
* A short term focus can signficantly change the outlook for a business - both upside and downside. But all of the experiments are short term - so business tend to ship things that look good in the short run and bad in the long - selection effect on mistakes. They never ship the things that look bad in the short term. So everything turns out worse than initially thought; adjustment is needed to expected effects. A/B tests are a gold standard test by test, but the system as a whole is biased.
* Low hanging fruit in the outcomes that are bad in the short term and good in the long run. If you can run a long term experiment, can show their effectiveness outside of the standard A/B framework. Up to the firm to provide incentives / rewards for capturing this value.
* Peer review for experimental design - making sure that the team running it has value alignment and metrics that are not gamed. 

### Recommendation Systems

* Netflix challenge - the utility matrix was sparse but entries missing for different reasons. If a movie just added, or some movies are never recommended, user wouldn't have had the chance to rate it. Decision to be made about how to treat missing values; setting dependent. In some settings not rated may be a negative rating, while in other settings empty cells are truly no interaction between user and item.
* We may have observable information about the items - the meta data of actors in a movie or genre etc. In S2M we will have the text of the story. Sometimes there is no additional benefit to having this data, we may instead strictly prefer user inputs.
* ML focused on prediction but some approaches also have interpretability. Sparse matrix approaches have real advantages over treating this with typical classification algos
* Article on EM algo - the text, item properties, point towards statistics researchers would be interested. From users who interacted with the article, we see vision researchers also like this article. Using latent dirichlet allocation to perform dimensionality reduction and assign topic interests to clusters.
* Might make sense to model topics on their own, outside of the user context. Creates defined contexts, user churn can confuse how items are related. In a more stable context, could more easily build in the user choices.
* Rec system could generate just predictions, but may want a bigger model with probabilistic predictions instead - important for S2M since we are running an experiment - need a prediction of how big the treatment effect should be.
* Typically assume pure additivity, no interaction models - too computationally expensive to consider combinations. Consider each item separately, not designing sets of recommendations.
*  Jaccard / cosine can be applied to item-item similarity (eg. text similarity via bigrams) but also user/item rating comparisons. Each item has a rating vector, so we can get an item-item similarity matrix from user ratings. In either case we can get an item-item similarity matrix for the recommendation step. Then to predict for a given user and test item, take a weighted average of their ratings for other items times the similarity of those items to the test item.

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

## Project Notes

### Recommendation Systems - Mining Massive Datasets Chapter 9

* Content-based systems examine properties of the items recommended. Collaborative filtering systems recommend items based on similarity measures between users and/or items.
* Utility matrix, gives for each user-item pair, a value that represents what is known about the degree of preference of that user for that item. The goal of a recommendation system is to predict the blanks in the utility matrix. It is only necessary to discover some entries in each row that are likely to be high. In most applications, the recommendation system does not offer users a ranking of all items, but rather suggests a few that the user should value highly.
* The long tail phenomenon: forces on-line institutions to recommend items to individual users. It is not possible to present all available items to the user, the way physical institutions can. Neither can we expect users to have heard of each of the items they might like.
* Utility matrix needs to be populated. Can ask users to rate items, but generally they are unwilling to rate many things. We can make inferences from their behavior - sort of a binary rating of engaging / buying product or not, but could scale with level of engagement (eg. viewing but not buying).

##### Content Matrix Methods

* In a content-based system, we must construct for each item a profile, which is a record or collection of records representing important characteristics of that item. Some will be boolean values and other real-valued, can still take cosine distance but might need some scaling so real valued do not dominate.
* Documents present special problems. First, eliminate stop words – the several hundred most common words, which tend to say little about the topic of a document. For the remaining words, compute the TF.IDF score for each word in the document. The ones with the highest scores are the words that characterize the document. 
  * To measure the similarity of two documents, there are several natural distance measures we can use: Jaccard or Cosine. For cosine, think of the sets of high TF.IDF words as a vector, with one component for each possible word. The vector has 1 if the word is in the set and 0 if not. 
* We also need vectors describing user preferences from the utility matrix. If the utility matrix is not Boolean, e.g., ratings 1–5, then we can weight the vectors representing the profiles of items by the utility value, normalizing by subtracting off the mean value. 
* With profile vectors for both users and items, we can estimate the degree to which a user would prefer an item by computing the cosine distance between the user’s and item’s vectors. The random-hyperplane and locality-sensitive-hashing techniques can be used to place (just) item profiles in buckets. 

##### Content Classification Algorithms

* A completely different approach to a recommendation system using item profiles and utility matrices is to treat the problem as one of machine learning - for each user, build a classifier that predicts the rating of all items.
* Unfortunately, classifiers of all types tend to take a long time to construct. For instance, if we wish to use decision trees, we need one tree per user.

##### Collaborative Filtering

* In place of the item-profile vector for an item, we use its column in the utility matrix. Instead of contriving a profile vector for users, we represent them by their rows in the utility matrix. Users are similar if their vectors are close according to some distance measure such as Jaccard or cosine distance. Recommendation for a user U is then made by looking at the users that are most similar to U in this sense, and recommending items that these users like.
* A larger (positive) cosine implies a smaller angle and therefore a smaller distance. Jaccard similarity  = IOU
* Distance measures across low and high ratings can cause some confusion in the metrics. Just treating high ratings as rated help clear up Jaccard and cosine confusion.
* If we normalize ratings, by subtracting from each rating the average rating of that user, we turn low ratings into negative numbers and high ratings into positive numbers. If we then take the cosine distance, we find that users with opposite views of the movies they viewed in common will have vectors in almost opposite directions, and can be considered as far apart as possible.
* The utility matrix can be viewed as telling us about users or about items, or both. There are two ways in which the symmetry is broken in practice: (1) We can base our recommendation on the decisions made by these similar users (2) it is easier to discover items that are similar because they belong to the same genre, than it is to detect that two users are similar because they prefer one genre in common, while each also likes some genres that the other doesn’t care for.
* Dually, we can use item similarity to estimate the entry for user U and item I. Find the m items most similar to I, for some m, and take the average rating, among the m items, of the ratings that U has given. As for user-user similarity, we consider only those items among the m that U has rated, and it is probably wise to normalize item ratings first. If we find similar users, then we only have to do the process once for user U. On the other hand, item-item similarity often provides more reliable information.
* Clustering - items or users. Our utility matrix is usually too sparse to use outright. Iteratively can slowly cluster items then users then repeat to reduce matrix. To predict for user U and item I, find their cluster, take that entry or use a filling method to approximate it.

##### Dimensionality Reduction

* Conjecture that the utility matrix is actually the product of two long, thin matrices. Makes sense if there are a relatively small set of features of items and users that determine the reaction of most users to most items.
* UV-decomposition:  from M (n x m) construct U (n x d) and V (d x m) st UV approximates M in those entries where M is nonblank. Closeness often measured by RMSE - summing nonblank M entries, taking square of entry-wise difference between M and UV
  * Computed incrementally - start with random initialized UV and move stochastically towards a minimum via grad descent. 