---
title: Introduction to Statistical Learning
date: 20191004
author: Spencer Braun
---



##### Chapter 3 - Linear Regression

Qualitative Predictors

* If it only has 2 possible values, then we can use a dummy indicator variable that takes on 1 or 0. If we coded a multi level indicator as (0, 1, 2), then Y would go up by beta from level 0 to 1 and another beta for 1 to 2, but this makes no sense for qualitative variables.
* If we pick 1 and -1 a simple linear regression, $B_0$ is the average value and I(1) indicates amount above the mean and I(-1) indicates amount below the mean for our subgroups
* For more than 2 levels for a variable, we simply create multiple true/false dummy variables. Always have one fewer dummy than the number of levels, since one can be the baselines when all dummies are false. Then we use the F test to determine whether the total of our dummies has significance beyond individuall p-values

Variable Interaction

* Can think of synergy in a relationship as akin to the interaction between variables. Since standard regression looks at the effect of one variable while holding others constant, we do not include interaction effects automatically
* Adding an interaction term $\Beta_3X_1X_2$ captures variable interaction. Good example of workers and production lines in a factory - to some extent both need to be expanded to increase capacity. We can interpret $β_3$ as the increase in the effectiveness of TV advertising for a one unit increase in radio advertising (or vice-versa).
* The **hierarchical principle** states that if we include an interaction in a model, we should also include the main effects, even if the p-values associated with principle their coefficients are not significant.

Non-Linear Relationships - Polynomial Regression: Can add a higher order polynomial term by taking a variable to a power. Still a linear model of terms, but capturing a non-linear relationship.

Common Problems 

* Non-linearity - useful to look at residual plots $e_i = y_i - \hat{y_i}$ vs $x_i$. Hope for the residual plot to be white noise, but the presence of a pattern indicates a potential problem with the model. 
* Correlated error terms will lead to underestimating the true SEs. Frequent problem with time series. We can see in plots of the residuals that they exhibit tracking, adjacent residuals may have similar values.
* Heteroscedasticity - non-constant variance of the error terms, ie. correlation between x and $\epsilon$. Often tranform Y to $log(Y)$ or $\sqrt{Y}$ .
* Outliers - have an unusual value of $y_i$ given $x_i$. High leverage points - have unusual value for $x_i$ and can have a sizable impact on the regression line. This is hard to identify in multiple regression settings just by eyeballing. We can calculate a leverage statistic: $h_i = \frac{1}{n} + \frac{(x_i - \bar{x})^2}{\sum_{i’=1}^n(x’_i - \bar{x})^2}$ . Always between 1/n  and 1 with average leverage over all points (p+1)/n.
* Collinearity - two or more predictors are closely related to one another. Reduces the accuracy of the estimates of the coefficients causing SE for coefficients to grow and the power of the hypothesis test is reduced. Useful to look at a correlation matrix of the variables, though if multicollinearity exists in the interaction of multiple variables this will not be caught. Compute the Variance Inflation Factor (VIF), ratio of variance of a beta when fitting the full model over the variance of beta if fit on its own. $VIF(\Beta_j) = \frac{1}{1 - R^2_{X_J | X_{-j}}}$ where $R^2_{X_J | X_{-j}}$ is the R^2 from a regression of $X_j$ onto all of the other predictors. If this term is close to 1, then you have collinearity and VIF will be large. 

Class Notes

* Matrix notation for a regression -> $y = X\beta + \epsilon$ where $y = (y_1 ... y_n)^T$ and $\Beta = (\Beta_1 ... \Beta_n)^T$ and X is our data matrix with an extra column of 1 for the intercept. X stores each variable in a column, so X is dimension n x (p+1) and B is dimension (p+1) vector. Betas are always linear but the predictors X do not have to be linear.
* Intuition behind F-test - $RSS_0$ is the RSS for the model under the null hypothesis where we set all of the betas to zero, just fitting the intercept. RSS is just the model RSS. $RSS_0$ is always bigger since predictors improve the fit of the regression. Comparing the two RSS stats, if the additionals betas don’t help much, then the RSS should be roughly the same. Otherwise we would expect a big improvement in the RSS over the replacement model with fewer betas. F-stat is the scaled ratio of the difference in the RSS over the RSS of the full model. In the specific case where we omit a single beta, then this is equivalent to a t-test for that beta.
* Multiple testing issue - Given 10 betas, could do 10 different t-tests, 1 for each beta. Over a large number of betas and at significance level of 5%, bound to find significance as a false positive. Over 10 tests, P(at least one is positive) = 1 - P(no false positives) = 1 - P(1st test accepts and 2nd...) = 1- P(1st test accepts)P(2nd test accepts)... = $1 - (0.95)^{10} = 0.40$
* p-value - if the null hypothesis is true, there is a p% chance of making a false positive, ie. rejecting $H_0$ in error.
* Stepwise approach - construct a sequence of p models and select the best among them. Performed through forward, backward, or mixed selection. Note for forward selection we are looking at the individual linear models to determine which betas to add at each step.
* Confidence interval vs prediction interval: they predict the same thing but CI is the output of $f(x_o)$ but the PI takes the full model $y = f(x_0) + \epsilon$, so it accounts for the additional error.