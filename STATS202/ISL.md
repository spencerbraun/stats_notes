---
title: Introduction to Statistical Learning
date: 20191004
author: Spencer Braun
---

[TOC]



### Chapter 3 - Linear Regression

##### Qualitative Predictors

* If it only has 2 possible values, then we can use a dummy indicator variable that takes on 1 or 0. If we coded a multi level indicator as (0, 1, 2), then Y would go up by beta from level 0 to 1 and another beta for 1 to 2, but this makes no sense for qualitative variables.
* If we pick 1 and -1 a simple linear regression, $B_0$ is the average value and I(1) indicates amount above the mean and I(-1) indicates amount below the mean for our subgroups
* For more than 2 levels for a variable, we simply create multiple true/false dummy variables. Always have one fewer dummy than the number of levels, since one can be the baselines when all dummies are false. Then we use the F test to determine whether the total of our dummies has significance beyond individuall p-values

##### Variable Interaction

* Can think of synergy in a relationship as akin to the interaction between variables. Since standard regression looks at the effect of one variable while holding others constant, we do not include interaction effects automatically. The *additive assumption* is the feature of linear regression that assumes you can treat the variables as having no interaction - a single variable has a fixed effect on the response variable, regardless of the other variables.
* Adding an interaction term $\Beta_3X_1X_2$ captures variable interaction. Good example of workers and production lines in a factory - to some extent both need to be expanded to increase capacity. We can interpret $β_3$ as the increase in the effectiveness of TV advertising for a one unit increase in radio advertising (or vice-versa).
* The **hierarchical principle** states that if we include an interaction in a model, we should also include the main effects, even if the p-values associated with principle their coefficients are not significant. This is akin to normalizing the variables being combined in the interaction term, since subtracting off the mean is the same as including a separate term for it. 

Non-Linear Relationships - Polynomial Regression: Can add a higher order polynomial term by taking a variable to a power. Still a linear model of terms, but capturing a non-linear relationship.

##### Common Problems 

* Non-linearity - useful to look at residual plots $e_i = y_i - \hat{y_i}$ vs $x_i$. Hope for the residual plot to be white noise, but the presence of a pattern indicates a potential problem with the model. 
* Correlated error terms will lead to underestimating the true SEs. Frequent problem with time series. We can see in plots of the residuals that they exhibit tracking, adjacent residuals may have similar values.
* Heteroscedasticity - non-constant variance of the error terms, ie. correlation between x and $\epsilon$. Often tranform Y to $log(Y)$ or $\sqrt{Y}$ . Typical to see in growth or compounding - think of regressions of volatility at HRT
* Outliers - have an unusual value of $y_i$ given $x_i$. High leverage points - have unusual value for $x_i$ and can have a sizable impact on the regression line. This is hard to identify in multiple regression settings just by eyeballing. We can calculate a leverage statistic: $h_i = \frac{1}{n} + \frac{(x_i - \bar{x})^2}{\sum_{i’=1}^n(x’_i - \bar{x})^2}$ . Always between 1/n  and 1 with average leverage over all points (p+1)/n.
  * If we think the outlier is an error, then we can remove it. But it could indicate misspecification of the model
  * If we have high leverage points for multiple predictors -> can make pairwise comparisons to see high leverage points between predictors. The leverage stat formalizes this idea for any dimension
* Collinearity - two or more predictors are closely related to one another. Reduces the accuracy of the estimates of the coefficients causing SE for coefficients to grow and the power of the hypothesis test is reduced. Useful to look at a correlation matrix of the variables, though if multicollinearity exists in the interaction of multiple variables this will not be caught. Compute the Variance Inflation Factor (VIF), ratio of variance of a beta when fitting the full model over the variance of beta if fit on its own. $VIF(\Beta_j) = \frac{1}{1 - R^2_{X_J | X_{-j}}}$ where $R^2_{X_J | X_{-j}}$ is the $R^2$ from a regression of $X_j$ onto all of the other predictors (Note $X_{-j}$ indicates all x excluding the jth predictor). If this term is close to 1, then you have collinearity and VIF blow up to be large. 
  * Collinearity causes coefficient estimates to become less certain. End up with exactly the same fit for mutliple betas - now difficult to optimize the model for each beta. Elongates contour lines of fit, meaning betas of highly varying degree are producing the same goodness of fit / CI for estimate. 

##### K-nearest neighbors regression (KNN regression)

* Given value K for prediction point $x_0$, identifies K training obervations closest to $x_o$, represented by $N_0$. It then estimates $f(x_0)$ using the average of all the training responses in $N_0$, ie. $\hat{f}(x_0) = \frac{1}{K}\sum_{x_i \in N_0} y_i$. 
* Value of K boils down to bias-variance tradeoff - small values of K more flexible with low bias and high variance. The parametric approach will outperform the nonparametric approach if the parametric form that has been selected is close to the true form of f. 
* We would not use KNN regression with a linear relationship; no matter how you choose K, you will always have higher test MSE than linear regression, since the linear model has almost no bias and also resists variance better. Linear regression can still outperform for non-linear relationships; may simply depend on the choice of K. For extreme non-linearities, then linear regression tends to underperform for wide range of K. 
* With small number of predictors, MSE for KNN remains somewhat constant over different f linearities while linear regression has much higher MSE for non-linear relationships. KNN can still underperform significantly with more predictors, however. Spreading 100 observations over p = 20 dimensions results in a phenomenon in which a given observation has no nearby neighbors—this is the so-called curse of dimensionality. 

##### Class Notes

* Matrix notation for a regression -> $y = X\beta + \epsilon$ where $y = (y_1 ... y_n)^T$ and $\Beta = (\Beta_1 ... \Beta_n)^T$ and X is our data matrix with an extra column of 1 for the intercept. X stores each variable in a column, so X is dimension n x (p+1) and B is dimension (p+1) vector. Betas are always linear but the predictors X do not have to be linear.
* Intuition behind F-test - $RSS_0$ is the RSS for the model under the null hypothesis where we set all of the betas to zero, just fitting the intercept. RSS is just the model RSS. $RSS_0$ is always bigger since predictors improve the fit of the regression. Comparing the two RSS stats, if the additionals betas don’t help much, then the RSS should be roughly the same. Otherwise we would expect a big improvement in the RSS over the replacement model with fewer betas. F-stat is the scaled ratio of the difference in the RSS over the RSS of the full model. In the specific case where we omit a single beta, then this is equivalent to a t-test for that beta.
* Multiple testing issue - Given 10 betas, could do 10 different t-tests, 1 for each beta. Over a large number of betas and at significance level of 5%, bound to find significance as a false positive. Over 10 tests, P(at least one is positive) = 1 - P(no false positives) = 1 - P(1st test accepts and 2nd...) = 1- P(1st test accepts)P(2nd test accepts)... = $1 - (0.95)^{10} = 0.40$
* p-value - if the null hypothesis is true, there is a p% chance of making a false positive, ie. rejecting $H_0$ in error.
* Stepwise approach - construct a sequence of p models and select the best among them. Performed through forward, backward, or mixed selection. Note for forward selection we are looking at the individual linear models to determine which betas to add at each step.
* Confidence interval vs prediction interval: they predict the same thing but CI is the output of $f(x_o)$ but the PI takes the full model $y = f(x_0) + \epsilon$, so it accounts for the additional error.
* Residual standard error: $RSE = \sqrt{\frac{1}{n-p-1}RSS}$ compensation for the increase in P increasing the the R^2
* R note: `lm(y ~ .)` regresses the y against all x terms in the dataset. 
* For higher dimensional residual analysis - plot the residuals against the $\hat{y}$ instead of x, look for a pattern. Pattern suggests a leftover structure to the y values that remains to be explained.
* Studentized residuals - $\hat{\epsilon_i} = y_i -\hat{y_i}$ is an estimate of the true epsilon. It has SE $\sigma\sqrt{1 - h_{ii}}$. Therefore we can divide the estimate by the SE. Should follow a t-distribution with n-p-2 DoF (Note OLS has residuals with mean 0, so we are only concerned with the estimate SE).

### Chapter 4 - Classification

* Bayes Classifier - P(Y|X) is known, then given input $x_0$ we predict the response $\hat{y_0} = argmax_y P(Y=y | X=x)$. Minimizes expected 0-1 loss $E\Big[\frac{1}{m}\sum_1^m 1(\hat{y_i} \neq y_i)\Big]$, ie a binary loss function of wrong or right. This is the best theoretical result.

##### Logistic Regression

* Mapping regression output to range of [0, 1]. To get a sigmoid from [0, 1], we use the **logistic function** $f(x) = \frac{e^x}{1 + e^x}$. For small x, close to 0 and close to 1 as $e^x$ grows.
* We can then generate log odds: $log\Big[\frac{P(Y=1|X)}{P(Y=0|X)}\Big] = \Beta_0 + \Beta_1X_1 + ... + \Beta_pX_p$. We cannot use least squares because we do not know the conditional probability - LHS not observed. Instead we use MLE.
* Confounding - one of the x variables explains another variable somewhat. Not collinearity, but a relationship exists between X parameters, eg. students are likely to have high card balances, people with high balances are more likely to default, given a high balance students are less likely to default. Running the regression with just student will give you a positive coefficient, and running with balance too will give a negative coefficient for student. In a simple logistic regression, the student is standing in for balance.
* Similar issues with collinearity - creates instability in estimating the coefficients and affects the convergence of the MLE fitting.

### Chapter 10 - Unsupervised Learning

##### Principal Components Analysis (PCA)

* When faced with a large set of correlated variables, principal components allow us to summarize this set with a smaller number of representative variables that collectively explain most of the variability in the original set.
* PCA seeks a small number of dimensions that are as interesting as possible, where the concept of interesting is measured by the amount that the observations vary along each dimension.
* The first PC of features $X_1,...,X_p$ is the normalized linear combination of the features $Z_1 = \phi_{11}X_1 + ... + \phi_{p1}X_P$ that has the largest variance. By normalized, we mean $\sum_{j=1}^p \phi^2_{j1} = 1$. The $\phi$ elements are the loadings of the first principal component, together the principal component loading vector.
* We look for the linear combination of sample feature values $z_{ij}$ , ie $\frac{1}{n}\sum_{i=1}^nz_{i1}^2$ subject to the constraint $\sum_{j=1}^p \phi^2_{j1} = 1$. The z-values are the scores of the first principal component. We assume that each of the X’s has mean zero. 
* The loading vector $\phi_1$ with elements $\phi_{11}, \phi{21},...,\phi{p1}$ defines a direction in feature space along which the data vary the most. If we project the n data points $x_1,...,x_n$ onto this direction, the projected values are the principal component scores $z_{11},...,z_{n1}$ themselves.
