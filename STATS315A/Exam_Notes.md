# Exam Notes

## Themes

* Scale invariance - how do units affect your method, standardizing predictors, effect on correlated predictor coefficients
* Correlation between predictors - how well identified are your coefficients / relationships. Correlation breeds uncertainty in model, regularization either discards correlated covariates or averages over them.
* Continuous vs discrete, blending of features in a reduced model or feature selection (eg. ridge / pcr vs lasso).
* p > n - now need some regularization. Consider when we perform regularization for variance reduction vs removing design matrix singularity
* Adding bias for variance reduction
* The SVD gives us many properties of a matrix. $A^TA=S$, eigenvalues of S are singular values of A. V are PCs, Ridge SVD shows it shrinks the smallest PCs the most. See page 66
* When do distributional assumptions kick in? Inference for linear regression, thinking about conditional vs joint densities for logistic vs LDA

## Chapter 2

* If the linear model is correct, or almost correct, K-nearest neighbors will do much worse than linear regression. (see bottom)
* Supervised problem can be an algorithmic approach of learning by example. Fitting black box to example after example. Statistically seen as a function approximation problem, estimating a form for f(X) (slide 30). Any function passing through all points has RSS 0, need to restrict the class of functions.
* Any method that attempts to approximate locally varying functions is “cursed”. Alternatively, any method that “overcomes” the curse, assumes an implicit metric that does not allow neighborhoods to be simultaneously small in all directions.
* KNN effective parameters - N/k, generally bigger than p. If the neighborhoods are non overlapping, there would be N/k neighborhoods and we would fit one parameter (mean) in each each neighborhood (15).
* Why look beyond KNN? We often have smaller samples, where linear inflexibility provides more stability. Curse of dimensionality makes the metric size of KNN explode - still converges but rate of covergence diminishes as dimension increase (19) As k -> infinity, the variance of the estimator ->0. As k/N -> 0, the bias -> 0.
* KNN and OLS replace expectation with averages over the training data, but OLS assumes linearity good approximation, KNN assumes locality is the best approximation.
* Curse of dimensionality - to capture same fraction of observations, need increasing volume of space. Median distance from origin to neighbor increases - data closer to boundary points. Sampling density inverse to dimensionality. Page (23)
* As dim increases, NN strays from the test point, say origin. As dimensionality increases, have to think about what this does to the prediction function. Can cause bias to increase if all predictions collapse to a single number, or variance to increase if tiny changes in X result in wildly different outputs. (24)
* We have to impose some complexity restrictions on the form of our model f(X), but no constraint avoids tradeoffs (33)

## Chapter 3

* When p > n, know that we must use a regularized regression or dimensionality reduction. Then consider do we want feature selection or reduction spread across predictors - lasso vs. PCR / Ridge
* Linear models can outperform flexible in low-data cases, low signal to noise cases, sparse data situations. The model is linear in the parameters no matter the form of X
* Hat matrix assumes full column rank of X - ie. rank p. It is the orthogonal projection of y onto the column space of X. If X singular, then beta not uniquely defined, but the regression is still the same projection (46)
* Testing for significance among groups of coefficients, say dummy vars for levels of a single measure -> F test. (48)
* The z scores measure the effect of dropping that variable from the model. Inclusion of correlated variables can turn a significant univariate coefficient into an insignificant multivar coefficient.
* While G-M theorem says OLS is the lowest var unbiased estimator, we often prefer to trade a little bias for more variance reduction (52). Most models are distorted, biased reflections of reality already.
* Orthogonalized regression - if Xp is highly correlated with some other Xk's, then zp will be near zero and beta-hat_p will be unstable. (55)
* We often can improve on least squares estimates in prediction accuracy (shrinking model coefs, reducing predictors) and interpretation (large p, care about the essential ones for interpretability) (57).
* We can always produce the forward stepwise model even for p >> n. Will have less variance but maybe more bias than best subset (59)
* Printing model after selection - the SEs are not valid since they do not account for the search process. Bootstrap can help estimate the SEs here (60). 
* Subset selection can increase variance since it is a discrete search. Shrinkage is more continuous and don't suffer from this variability (61)
* With correlated variables in regression, coefficients are poorly determined and exhibit high variance. Ridge penalty constrains size of coefficients, alleviating this problem. (One cannot shoot to infinity to be offset by the other) (63)
* Scaling - PCR, Ridge, Lasso, PLS all scale dependent, so usually standardize. Ridge designed to fix singularity problem of X. PCR and Ridge are linked methods - see (66). Ridge shrinks the smallest PCs the most - the directions with minimum variance. Implicitly assume that the response will vary most in the same directions as the inputs. Ridge regression shrinks the coefficients of the PCs, shrinking more depending on the size of the corresponding eigenvalue; PCR discards the p-M smallest eigenvalue components. (79)
* Lasso - soft thresholding, the translation of each coefficient by a constant fact lambda, truncating at zero. (As ooposed to hard thresholding of best subsets). 
* PLS - produces a sequence of derived, orthogonal inputs or directions z1,...,zM. M <p reduces the regression (80). Seeks directions with high variance and high correlated with y, but variance tends to dominate and PLS acts similarly to ridge and PCR.
  * Univariate regressions for all X. $\mathbf{z}_{1}=\sum \hat{\gamma}_{\ell} \mathbf{x}_{\ell}$ first PLS direction. `lm(y ~ z_1)` -> beta_hat 1. Orthogonalize y, x wrt z: r1 = y - beta1 z1, $\mathbf{x}_{\ell}^{*}=\mathbf{x}_{\ell}-\hat{\theta}_{\ell} \mathbf{z}_{1}$, repeat. Produces sequence of derived directions z.
* Comparison of methods with correlated predictors on page 82.
* Regression coefficient βj estimates the expected change in y per unit change in xj , *with all other predictors held fixed*. But predictors usually change together! (Slide 22)
* CV: Note that different subsets of size λ will (probably) be found from each of the K cross-validation training sets. Doesn’t matter: focus is on subset size, not the actual subset. The focus is on *subset size*—not which variables are in the model.
* *Ridge vs Lasso:* if inputs are orthogonal, ridge *multiplies* least squares coefficients by a constant < 1, lasso *translates* them towards zero by a constant, truncating at zero. (Slide 53)

## Chapter 4

* Linear Regression on an indicator matrix - can view the regression as an estimate of conditional expectation of Y given X, ie. G = k given X = x. (104) While fitted function might sum to 1, may not be a true probability due to domain. Can still work well with basis function expansion. Also the target approach is equivalent - target is a column of one hot matrix - and we aim to reproduce appropriate target for observation.
* Masking - when K, number of classes, large classes can be masked by others. Regression misses the linear separation (105). Could be fixed with polynomial, and degree up to K-1 might be needed to fix masking.
* LDA - many class density methods, QDA, LDA, non parametric densities, naive bayes. LDA assumes common covariance across classes, QDA separate for each class (108). LDA's direction does not depend on Gaussian, but its particular cut point does. (110)
* Both LDA, QDA have good track record even without normal data. The data may only support simple decision boundaries and the estimates provided by the Gaussian models are stable. We put up with the bias of a linear boundary bc it has much lower variance than exotic alternatives.
* We can regularize QDA like ridge regression - shrink the covariance matrices towards LDA. (112)
* Reduced rank LDA - using the PCs of the centroids. Good when centroids are mostly clustered around a lower dimension. Summary on 116. 
* Logistic regression fit on SA heart data - could drop the least significant coefficient and refit the model repeatedly until no further terms could be dropped. Better yet, refit each of the models with one variable removed, then perform an analysis of deviance to decide which variable to exclude. (124)
* Interpreting logistic coefficients - coef of 0.8 -> increase of unit of Xi accounts for an increase in the odds of logistic 1 outcome of exp(0.8) = 1.084 -> 8.4% (See 124 for SEs also)
* MLE parameter estimates beta hat are the coefficients of a weighted least squares fit. The weighted RSS is the pearson chi square statistic (quadratic approx of deviance), can apply CLT, show beta consistency
* LDA and logistic have same form, but the way the coefs estimated diffs them. Logistic is more general with fewer assumptions - leaves the marginal density of X as arbitrary and maximizes the conditional likelihood. LDA max's the full log likelihood of the joint (127). Leads LDA to have lower variance since we know more about the parameters; if we actually have gaussians then we get lower efficiency. But LDA is not robust to gross outliers since incorporates information about all of the data.
* LDA direction: (skew projection) Although the line joining the centroids defines the direction of greatest centroid spread, the projected data overlap because of the covariance (left panel). The discriminant direction minimizes this overlap for Gaussian data (right panel). (slide 10)
* Logistic with p >>n: model now has to be regularized with ridge (similar to SVM) or lasso selected variables. Path algorithms can do efficiently. (Slide 57)
* Discriminative vs generative (informative) learning: logistic regression uses the conditional distribution of Y given x to estimate parameters, while LDA uses the full joint distribution (assuming normality).
* Naive Bayes - uses Bayes formula for P(Y=k|X) given class densities. Assumes within class densities (ie, distribution among predictors) independent for each predictor. (Slide 60)

## Chapter 5

* Linear models are convenient (easy to interpret, Taylor approximations of f(x)) and sometimes necessary with N small and p large. (139). Basis expansion allows us to apply this in a more flexible way
* M order spline is a piecewise polynomial of order M, continuous derivs up to M-2 (cubic spline M=4). (144) Often parametrize in terms of df and let x's determine the knot locations.
* For smoothing splines, just selecting a penalty lambda since we have knots at every data point (158). More on bias variance, overfit and underfit on 160.
* B-spline basis

## Chapter 18 

* p >>n - worry about high variance and overfitting. Can overfit even with a linear model. Often go with a simple, highly regularized approach (649)
* Simulated ridge example - with lower p, lower regularization can have best test error. At large p, want high regularization (higher df). When p = 1000, even though there are many non zero coefs, we don't have a hope for finding them and we need to shrink all the way down. 
* Cannot estimate the high dimension covariance matrix so ridge cannot exploit the correlation among predictors. High penalty then leads to better prediction performance.
* Diagonal LDA assumes independence among features in a class, special case of naive bayes and equivalent to nearest centroid.(652) But uses all of the features so does not perform selection and interpretation is more difficult.
* Nearest shrunken centroid - shrink the classwise mean toward the overall mean for each feature separately. (652) Replace centroids in discriminant score with shrunken cousins. Vast majority of predictors are discarded since only the predictors with a nonzero value after soft-thresholding play a role in the classification rule. (see ch 4 slide 22) Choose the shrinkage factor by CV. Denoises large effects, sets small ones to zero. With more than two classes, performs gene selection and different number of genes for each class.
* RDA - shrink shared covariance matrix towards its diagonal. Perfectly shrunk to diagonal is then equivalent to diagonal LDA and nearest (non shrunken) centroid. Can also regularize logistic regression (657)
* SVD shortcuts computation - when p > N, computations can be carried out in N dimensional space using N x N D and p x N V. Results can be generalized to all models that are linear in parameters with quadratic penalties (659)
* Using L1 penalties - when p > N, the number of non zero coeffients is at most N for all values of lambda. Coordinate descent efficient algorithm for L1 reg. logistic regression. (661)
* Elastic net - second term encourages highly correlated features to be averaged, while the first term encourages a sparse solution in the coefs of these averaged features (662). Ridge is averaging over correlated features.
* Large lasso - via coordinate descent: optimize each parameter seaprately holding others fixed. Updates are trivial, cycle until stable. SLS slide 10

## Other 

* Bootstrap - when we aren't sure about SEs, confidence intervals. If we treat X as random, may not get an analytical solution.

![Screen Shot 2020-02-17 at 2.40.26 PM](/Users/spencerbraun/Documents/Notes/Stanford/STATS315A/Screen Shot 2020-02-17 at 2.40.26 PM.png)

* OLS: $EPE(X_0) \approx \sigma^2 \times \frac{P}{N}\sigma^2 + bias^2_{OLS}(x_0)$
  * How did we get P/N: $\hat{\beta} = (X^TX)^{-1}X^Ty$ and $\hat{f} = X\hat{\beta}$. Say $Y \sim f(x) + \epsilon$ for $\epsilon \sim (0,\sigma^2)$. Treat X’s in training sample as fixed, condition on those X’s, then $\hat{f} = X\hat{\beta}=Ay$. $Cov(Y) = \sigma^2I_n$ so $Cov(\hat{f}) = \sigma = A\sigma^2$. What is the avg variance of $\hat{f}_i$? $Var(\hat{f}_i) = tr(A)\sigma^2/n$ where $tr(A) = p =$ # of parameters. So this is close to P/N sigma squared
* KNN: $EPE(X_0) \ge \sigma^2 + \sigma^2 + bias^2_{1-NN}(x_0)$
  * Sigma squared - both irreducible error and the error from the closest neighbor. 
* Concluding $\sigma^2 + P\sigma^2/N \approx \sigma^2 < 2\sigma^2$ - OLS better on variance. f(x) determines the bias, but for large N KNN could be optimal.

![Screen Shot 2020-02-17 at 3.07.33 PM](/Users/spencerbraun/Documents/Notes/Stanford/STATS315A/Screen Shot 2020-02-17 at 3.07.33 PM.png)