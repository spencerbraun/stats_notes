[TOC]

# Causal Inference Articles



## Recommendations as Treatments: Debiasing Learning and Evaluation. Schnabel et al.

* Viewing recommendation from a causal inference perspective, we argue that exposing a user to an item in a recommendation system is an intervention analogous to exposing a patient to a treatment in a medical study.
* Let binary matrix O represent which movies users provided ratings for. Define $\hat{Y}$ to encode recommendations as a similar binary matrix like O, limited to budget of k recs per user.
* A reasonable way to measure the quality of a recommendation is the Cumulative Gain (CG) that the user derives from the recommended movies, which we define as the average star-rating of the recommended movies in our toy example
* We are faced with the counterfactual question: how well would our users have enjoyed themselves (in terms of CG), if they had followed our recommendations
* To get unbiased estimates of recommendation quality despite missing observations, consider the following connec- tion to estimating average treatment effects of a given pol- icy in causal inference, that was already explored in the contextual bandit setting
* We refer to Pu,i as the *propensity* of observing Yu,i. In the *experimental* setting, we know the matrix P of all propensities, since we have implemented the assignment mechanism. In the *observa- tional* setting, we will need to estimate P from the observed matrix O. 
* Use inverse propensity weighting in the experimental setting. For observation settings, we merely need estimated propensities that are “better” than the naive assumption of observations being revealed uniformly. 
* Propensity Estimation via Naive Bayes: assuming that dependencies between covariates X, Xhid and other ratings are negligible.
* Propensity Estimation via Logistic Regression: aims to find model parameters φ such that O be- comes independent of unobserved X hid and Y. 

## The Deconfounded Recommender: A Causal Inference Approach to Recommendation. Wang, Liang, Charlin, Blei.

* A causal approach to recommendation, one where watching a movie is a “treatment” and a user’s rating is an “outcome.” The problem is there may be unobserved confounders, variables that affect both which movies the users watch and how they rate them.
* Frame the recommendation problem as an intervention: how would you rate this movie if forced to watch it. The traditional approach would only work under randomized movie watching and rating. There are many confounders - can think of a director the user likes as a confounder.
* Deconfounded recommender: we have two data streams what movies were watched and how they were rated by a user. The *exposure* data comes from a model by which users discover movies to watch; the *ratings* data comes from a model by which users decide which movies they like. The ratings data entangles this information. 
* The decounded recommender tries to correct the bias: uses the exposure data to estimate a model of which movies each user is likely to consider then uses this exposure model to estimate a substitute for the unobserved confounders. Then it fits a ratings model (e.g., matrix factorization) that accounts for the substitute confounders
* Matrix factorization as potential outcomes: for user u and movie i, with potential outcomes $\left(y_{u i}(0), y_{u i}(1)\right)$ for unseen and seen movies. We can separate movie ratings into a binary watched dataset and a numerical rating set. The system predicts each user’s ratings for their unseen movies and uses these estimates to suggest movies to each user - ie. prediction under intervention, what would you rate if forced to watch movie i.
* Adjusting for confounders: the prior approach assumed ignorability of confounders, which seems unlikely to hold. We need to control for confounders $w_{u}$ st $\left\{\boldsymbol{y}_{u}(0), \boldsymbol{y}_{u}(1)\right\} \perp \boldsymbol{a}_{u} | w_{u}$. This requires strong ignorability, controlling for all confounders. Not testable.
* Deconfounded approach: employes multiple causal inference since multiple treatments. Each binary movie exposure is a treatment so $I$ treatments, and I-vector outcome of seen or unseen per user. Fit Poisson factorization model to the exposure data since works well with binary data fit with coordinate ascent. With a fitted PF model, the deconfounded recommender computes a substitute for unobserved confounders. To form recommendations, calculates all potential ratings from model and orders the unseen movie potential ratings.
* Focusing on PF as the exposure model, we can also use other outcome models, e.g. Poisson matrix factorization and weighted matrix factorization

## Estimation Considerations in Contextual Bandits. Dimakopoulou, Zhou, Athey, Imbens

* We develop parametric and non-parametric contextual bandits that integrate balancing methods from the causal inference literature in their estimation to make it less prone to problems of estimation bias. Our methods aim to balance covariates between treatment groups and achieve contextual bandit designs which are less prone to problems of bias. We establish theoretical guarantees of BLTS and BLUCB that are comparable to LinTS and LinUCB.
* Contextual bandits seek to learn a personalized treatment assignment policy in the presence of treatment effects that vary with observed contextual features.
* In the initial phases of learning when samples are small, biases are likely to arise in estimating the outcome model using data from previous non-uniform assignments of contexts to arms. When the heterogeneity in treatment effects is too complex to estimate well with small datasets, when training observations from certain regions of the context space are scarce.
* Balancing will lead to lower estimated precision in the reward functions, and thus will emphasize exploration longer than the conventional linear TS and UCB algorithms, leading to more robust estimates.
* Create balanced versions of linear Thompson sampling (LTS) and linear UCB, demonstrating empirical outperformance for these methods. BLTS outperforms all other methods, giving wider confidence bounds.
* Doubly-robust: robust against misspecification of the reward function, more important here, and robust against the specification of the propensity score 
* For parametric estimation, use GLMs to model $\mu_a = \mathbb{E}[r(a) | x]$, say using LASSO or ridge. Bootstrap used to get sampling distributions. Train on many bootstrap samples drawn from X and r for given action to estimate sampling distribution $\mu_a(x)$.
* For non-parametric, estimate via random forest on history of observations assigned to action a.
* TS / UCB assignment rules for contexts - until every arm has been pulled at least once, the first contexts are assigned to arms in A at random with equal probability.
* Perform balancing via inverse propensity weighting (IPW). For UCB, we train a multi-class logistic regression model of **a** on **X** to estimate the assignment probabilities pa(x), a ∈ A, also known as propensity scores. For Thompson sampling, the propensity scores are in principle known because Thompson sampling performs probability matching. Then the weight for each observation (x, a, r) is $w_{a}=1 / \hat{p}_{a}(x)$. 
* Weighting the observations by the inverse propensity scores reduces bias, but even when the propensity scores are known it increases variance, particularly when they are small. 
* Balancing has a significant impact on the performance of UCB, since BLUCB finds the optimal assignment after 110 observations, much faster than LinUCB. 
* In real-world domains, the true data generative process is complex and very difficult to capture by the simpler outcome models assumed by the learning algorithms. Hence, model mismatch is very likely. For misspecied models, BLTS is able to harness the advantages of the stochastic assignment rule of Thompson sampling and find the optimal arm much more quickly. BLUCB does not handle better than LinUCB the estimation problem created by the deterministic nature of the assignment in the mis-specified case.
* Classification: The classifier can be seen as an arm-selection policy and the classification error is the policy’s expected regret. If only the loss associated with the policy’s chosen arm is revealed, this becomes a contextual bandit setting.
* Lasso vs Ridge Contextual Bandits: In the initial learning observations, a ridge bandit, due to L2 regularization, brings in all of the nuisance and noise contextual variables. The nuisance contextual variables affect assignment (possibly in non-linear ways) and act as confounders - increases variance of estimation.
* The generalized random forest bandit outperforms the LASSO and the ridge bandits. In cases where the outcome functional form is complicated, which is expected in real-world settings, bandits based on non-parametric model estimation may be proven useful and perform better, but also has a more complex assignment model and the sparsity of the LASSO is an advantage. The inability of the LASSO bandit and the ability of the generalized random forest bandit to fit the potential outcome model of the third arm, results in a strong performance edge of the latter.

## Machine Learning: An Applied Econometric Approach. Mullainathan, Spiess

* Many economic applications revolve around *parameter estimation*, while ML is focused on prediction problems often flexible and complex in nature.
* Often a modeling task first involves a prediction task. For example, the first stage of a linear instrumental variables regres- sion is effectively prediction. The same is true when estimating heterogeneous treatment effects, testing for effects on multiple outcomes in experiments, and flexibly controlling for observed confounders.
* Economic theory and content expertise play a crucial role in guiding where the algorithm looks for structure first. This is the sense in which “simply throw it all in” is an unreasonable way to understand or run these machine learning algorithms.
* Having a reliable estimate of predictive performance is a nonnegotiable requirement for which strong econo- metric guarantees are available. *Firewall principle*: none of the data involved in fitting the prediction function—which includes cross-validation to tune the algorithm—is used to evaluate the prediction function that is produced. 
* Even when machine-learning predictors produce familiar output like linear functions, forming these standard errors can be more complicated than seems at first glance as they would have to account for the model selection itself. Using the LASSO, repeated runs chooses very different covariates on different folds. The predictions are largely the same but predictor correlation prevents us from saying something meaningful about which predictors are most important.
* ML can use new sources of data, particularly relevant where reliable data on economic outcomes are missing, such as in tracking and targeting poverty in developing countries 
* Prediction in the service of estimation in IV: the finite-sample biases in instrumental variables are a consequence of overfitting. This biases the second stage coefficient estimate towards the OLD result. Overfit will be larger when sample size is low, the number of instruments is high, or the instruments are weak. Can use CV, jackknife, LASSO, Ridge for first stage prediction problem. 
* Even when there appears to be only a few instruments, the problem is effectively high-dimensional because there are many degrees of freedom in how instruments are actually constructed - linearly, logarithmically, dummies, interactions. 
* Propensity scores - see Lee, Lessler, and Stuart (2010) and Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, and Newey (2016). 

