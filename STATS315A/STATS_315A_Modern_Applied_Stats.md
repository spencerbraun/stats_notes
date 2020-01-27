[TOC]

# STATS 315A

## Chapter 1: Motivating Examples

* Prostate Cancer Dataset
  * Pairwise comparisons - can see binary vars like svi, gleason takes 4 values but only a single observation for value 3
  * For variables with zero values but non negative, take $log(x + \epsilon)$
  * Could split a variable - zero or non zero, then have the values for the non-zero portion as a new variable
  * Can see outliers in pairwise, investigate errors say in data entry
* Phoneme Classification
  * Discretize the analog spoken data via sampling, trying to classify sounds as “aa” or “ao”
  * For each observation have feature vector $x = (x_1,...,x_{256})$. Features are the values of the periodogram at each index.
  * Using logistic regression - take log odds: $log(\frac{P(v = aa)}{P(v=ao)}) = \beta_0 + \sum_{j=1}^{256}x_j\beta_j$. Fit the model via MLE to the data, can plot the estimates as a function of frequency. The estimates are variable - have many parameters and not a lot of data.
    * We sampled at 256 data points per analog curve - some arbitrariness, could have sampled at different frequency. 
    * Correlation among the features - adjacent samples are coming from adjacent / similar parts of the curve. The higher the resolution in the sample the closer the values - expect collinearity in our samples, the autocorrelation means the coefficients are not very well determined.
    * Red curve - could introduce some regularization for coefficient sequence. For example constraint $\sum_{j=1}^{255}|\beta_j - \beta_{j+1}| \leq c$.
    * Better methods such as filtering will be introduced in chapter 5. Once we constrain the coefficients to be smooth, our sampling frequency no longer makes much of a difference and our correlation problem is not so large
* Handwritten digits - similar to phonemes, taking an analogue signal, choosing a resolution to discretize the data
  * With no information, could take the frequencies of the digits in a training set, predict the mode for each OOS. Null error rate - misclassification rate with naive guessing. With 10 digits, null rate is 10% accuracy
  * Linear methods were troubled by the variations in writing digits. KNN just needs to find the example that is closest to the OOS observation at hand. KNN works very well in many classification settings.
* Tumor Genetic Markers
  * Wide dataset over 20k genes. ORganized into a matrix where each column is a sample, each row is a gene. Heat map, red overexpressed, green underexpressed. Hierarchical clustering leads to the groupings of red and green. Many more categories than observations so have to be careful about overfitting.
* Land Usage Map
  * Use another KNN type method. Take a pixel and the 8-neighborhood - the squares in a grid around it. Then shift the target pixel, build up big data set of 36-vectors corresponding to each pixel in the image. Then in 36-dimensions, predict the land usage based on NN in training set.

## Chapter 2: Supervised Learning

* Goals to predict test cases, understand the inputs effect on the outcome, assess the quality of predictions and inferences
* Confidence could be derived from distance from decision boundary in classification. For regression, prediction interval smaller in center in dataset where we have many training observations, but a test point near the edge of our training set have much larger error bands. In higher dimensions, have all kinds of holes in the feature space. 
* Netflix prize - somewhere in between supervised, unsupervised due to data sparsity. Led to development of matrix completion  - SVD / PC to construct a low rank matrix approximation. X mxn - A mxk B kxn. $min_{A,B}||X - AB||_F^2$, min over Frobenius norm. $X = UDV^T$ and m >> n, then $U_{m \times n},D_{n\times n}, V^T_{n \times n}$ - the D take values of 0 after we reach the rank k, then can distribute the orthogonal matrices into A and B and reconstruct. Problems? We did not handle the NAs and SVD will be performed on huge matrix.  Can actually solve the problem with iterating least squares - given an A, our minimization is a least squares problem, then given B, solve for A. 
  * To solve the missing values problem, can redefine the norm $min_{A,B}||X - AB||_{F,\Omega}^2$. Could try performing a weighted least squares $\sum_{i=1}^m w_i(x_{ij} - a_i^Tb_j)^2$, setting missing values to weights 0. Perform for each column of x, not efficient since have to rerun for each movie.
  * Simpler idea: fill in NAs with some guess, then use SVD to find an A and B. Then use the A and B to fill in the missing values with better values. Then repeat - called hard impute. Not guaranteed to finda global minimum but get some improvement. 
  * Could store the original matrix in a sparse matrix format so it takes up little space. $\Omega$ is just where we have observed values - we can make a matrix of binary values where values exist - then $P_{\Omega}(X)$ gets rid of NA values. $P_{\Omega^{\perp}}(AB)$, where $P_{\Omega^{\perp}}$ is the opposite of omega, where all are missing. Take $P_{\Omega}(X)- P_{\Omega}(AB) + AB$. The first two terms are sparse and AB is low rank - both stored easily.

### Notation

* X: Input variable. If X is a vector, its components can be accessed by subscripts $X_j$
* Quantitative outputs will be denoted by Y , and qualitative outputs by G
* Observed values are written in lowercase; hence the ith observed value of X is written as $x_i$
* Matrices are represented by bold uppercase letters; for example, a set of N input p-vectors $x_i$, i = 1,...,N would be represented by the N ×p matrix $\bold{X}$. 
* Since all vectors are assumed to be column vectors, the ith row of X is $x^T_i$ , the vector transpose of $x_i$.

### Least Squares

* Given vector of inputs $X^{T}=\left(X_{1}, X_{2}, \ldots, X_{p}\right)$, we predict Y with the model $\hat{Y}=\hat{\beta}_{0}+\sum_{j=1}^{p} X_{j} \hat{\beta}_{j}$. $\hat{\beta}_{0}$ is the intercept or bias.
* Including a constant variable 1 in X and $\hat{\beta}_{0}$ in the vector of coefficients $\hat{\beta}$, we can write the model as the inner product: $\hat{Y}=X^{T} \hat{\beta}$. (Make the inclusion assumption moving forward). $X^T = \left[1 X_1X_2\right]$
* Here $\hat{Y}$ is a scalar since we model a single output, though $\hat{Y}$ could be a K-vector making $\hat{\beta}$ a p x K matrix.
* Least Squares minimizes RSS: $\operatorname{RSS}(\beta)=\sum_{i=1}^{N}\left(y_{i}-x_{i}^{T} \beta\right)^{2} = \|y-X \beta\|^{2} = (\mathbf{y}-\mathbf{X} \beta)^{T}(\mathbf{y}-\mathbf{X} \beta)$
  * X is an N x p matrix with each row an input vector, y is an N-vector of the outputs of the training set
* Normal Equations: differentiate wrt $\beta$ to get $\partial \mathbf{RSS}/ \partial \beta=-2 X^{T}(y-X \beta)=0\ = {X}^{T}({y}-{X} \beta)=0$
* For non-singular $\mathbf{X}^{T} \mathbf{X}$, the unique solution given by $\hat{\beta}=\left(\mathbf{X}^{T} \mathbf{X}\right)^{-1} \mathbf{X}^{T} \mathbf{y}$, with specific point prediction given by $\hat{y}\left(x_{i}\right)=x_{i}^{T} \hat{\beta}$
* Geometric interpretation
  * $\hat{Y} = X\hat{\beta}$ is the orthogonal projection of y onto the subspace $M \subset \R^n$ spanned by the columns of X. (True even if X is not full rank). 
* We seldomly actually invert $X^TX$ in practice - instead use QR decomposition 
  * For full rank: $\begin{aligned}
    \|y-X \beta\|^{2} &=\left\|Q^{T} y-R \beta\right\|^{2} 
    =\left\|Q_{1}^{T} y-R_{1} \beta\right\|^{2}+\left\|Q_{2}^{T} y\right\|^{2} 
     \Rightarrow \hat{\beta}=R_{1}^{-1} Q_{1}^{T} y \\
    \end{aligned}$ and $\operatorname{RSS}(\hat{\beta}) =\left\|Q_{2}^{T} y\right\|^{2}$, $e=Q^{T} y$
  * $\hat{y}=Q_{1} Q_{1}^{T} y=H y=X\left(X^{T} X\right)^{-1} X^{T} y$ - H is the hat matrix bc it puts the hat on y
* For classification, could create a linear boundary with regression. If training data comes from bivariate Gaussians this model is pretty much optimal save for a few tweaks (which is somewhat surprising!), but if each class comes from a mixture of 10 low-variance Gaussians mixed together, need something more flexible - such as KNN.
* For mixed Gaussians $f(x) = \sum_{j=1}^{10}\pi_j\phi(\mu_j,\Sigma_j)$ where $\sum_{i=1}^{10}\pi_j =1$ - ie a weighting of the various Gaussians with different means and covariances $\Sigma_j=\sigma^2I$. Have means scattered with circular distributions around them - they mix together quite a bit. Flexible way of generating densities - nice for generative models, since have specified means, could be good for generating clusters.

### Nearest Neighbors

* Memory based method, no model, just fit based on stored training data.
* Drawn decision boundary - contour drawn where blue and orange classifications split for discrete test points. With 1-NN, decision boundary drawn via Voronoi tesselation, the boundary bisects the distance between each pair of different class points.
* 1-NN can certainly overfit, but it many problems like image classification, 1-NN can do really well when the signal to noise ratio is really high.
* KNN model: $\hat{Y}(x)=\frac{1}{k} \sum_{x_{i} \in N_{k}(x)} y_{i}$. Here $N_{k}(x)$ is the neighborhood of x defined by the k closest points (euclidean distance) to xi in the training sample. Essentially just averaging over closest points.
* For K=1, each point $x_i$ has a tile bounding the region for which it is the closest input point. The error on the training data will always be 0 for K=1
* While KNN appears to have a single parameter K, the effective number of parameters if N/k > p from least squares, decreasing with increasing k. Could imagine a one dimensional distribution of data, place them into bins and take majority votes within bin - then we are specifying N/k bin votes, which are our effective parameters
* Kernel smoother - instead of bins imagine an interval around the target test point, take a smooth weighting function over the points within the interval so that farther training points get a continuously smaller vote. Kernel smoothing differs from local linear regression most where data has a stronger trend than an average of the points in the region.
* Test set of 10k points - make simulated test sets huge to get a better reading of your method. No reason to stay with a small test set that will have its own variance problems.
* Optimal Bayes Error rate  - can write a joint distribution since we simulated the data. Then use Bayes rule to work out the exact conditional probability: $P(Y=1, X=x)=\frac{1}{2}f_1(x),\;P(Y=0, X=x)=\frac{1}{2}f_0(x)$, then $P(Y=1|X=x) = \frac{\frac{1}{2}f_1(x)}{\frac{1}{2}f_1(x) + \frac{1}{2}f_0(x)} = \frac{f_1(x)}{f_1(x) + f_0(x)}$  

### Statistical Decision Theory

* $X \in \mathbb{R}^{p}$ - a real valued random input vector. $Y \in \mathbb{R}$ - real valued output vector. Joint distribution P(X,Y)
* Seek a function $f(x)$ for predicting Y from X, and loss function L(Y, f(x)) for penalizing errors in prediction
* Using squared error loss, $\operatorname{EPE}(f)=\mathrm{E}(Y-f(X))^{2}=\int[y-f(x)]^{2} \operatorname{Pr}(d x, d y)= E_{X Y}(Y-f(X))^{2} =\mathrm{E}_{X} \mathrm{E}_{Y | X}\left([Y-f(X)]^{2} | X\right)$. We convert the squared error loss to an optimization objective. Note $$E_{X}\left(E_{Y | x}(Y-E(Y | x))^{2}\right.$$ does not contain f(x) - it is the irreducible error that does not depend on the form of f. (Had we used the absolute loss, we would get the conditional median instead of mean).
* Minimizing EPE point wise, $f(x)=\operatorname{argmin}_{c} \mathrm{E}_{Y | X}\left([Y-c]^{2} | X=x\right)$, we get $f(x)=\mathrm{E}(Y | X=x)$
* The final conditional expectation is known as the regression function. Thus the best prediction of Y at any point X = x is the conditional mean, when best is measured by average squared error.
* With KNN, we use $\hat{f}(x)=\operatorname{Ave}\left(y_{i} | x_{i} \in N_{k}(x)\right)$ - we approximate the expectation by an average and conditioning at a point relaxed to conditioning on a region “close” to the target point. This average becomes more stable as N grows, converges to expectation.
* We often do not have enough data for convergence or p grows too large, slowing the rate of convergence
* For regression, $f(x) \approx x^{T} \beta$, then  $\beta=\left[\mathrm{E}\left(X X^{T}\right)\right]^{-1} \mathrm{E}(X Y)$. The least squares solution amounts to replacing these expectations with averages over the training data.
* Using L1 loss instead of L2, the solution becomes $\hat{f}(x)=\operatorname{median}(Y | X=x)$ - more robust than conditional mean.
* Classification
  * Loss function K x K matrix L, zero on diagonal and non negative elsewhere. $L(k, \ell)$ is price paid for assigning l to a k observation - loss matrix k x k.
  * Using 0-1 loss, all misclassifications are charged a single unit: $\mathrm{EPE}=\mathrm{E}[L(G, \hat{G}(X))] = \mathrm{E}_{X} \sum_{k=1}^{K} L\left[\mathcal{G}_{k}, \hat{G}(X)\right] \operatorname{Pr}\left(\mathcal{G}_{k} | X\right)$
  * Solved by $$\hat{G}(x)=\operatorname{argmin}_{g \in \mathcal{G}} \sum_{k=1}^{K} L\left(\mathcal{G}_{k}, g\right) P\left(\mathcal{G}_{k} | X=x\right)$$ or can think of the posterior as  $\hat{G}(x)=\mathcal{G}_{k} \text { if } \operatorname{Pr}\left(\mathcal{G}_{k} | X=x\right)=\max _{g \in \mathcal{G}} \operatorname{Pr}(g | X=x)$ - the Bayes classifier. Error rate of the Bayes classifier is called the Bayes rate
  * Conditioning on x, say in 2 dimensions, we look at the one dimensional distribution of Y above that x. Our best estimate then is the mean of that one dimensional distribution of Y. For classification, we then just look at the probabilities of the class labels, and with 0-1 loss, select the class with highest probability.
  * This is all fine given a known distribution. With KNN, since we don’t know the distribution of Y given any x, we relax these expectations and assume the density is locally smooth in a small neighborhood around the target point.

### Local Methods in High Dimensions

* Curse of dimensionality - try to capture fraction r of observations, edge length of volume needed given by $e_{p}(r)=r^{1 / p}$. For even moderately large p, we have to capture the majority of the range of the each input variable. Secondly, most data points are closer to the boundary of the sample space than to any other data point. Thirdly, the sampling density is proportional to $N^{1/p}$, where p is the dimension of the input space and N is the sample size - in high dimensions all feasible training samples sparsely populate the input space.
* MSE at a point: $\begin{aligned}
  \operatorname{MSE}\left(x_{0}\right) &=\mathrm{E}_{\mathcal{T}}\left[f\left(x_{0}\right)-\hat{y}_{0}\right]^{2} 
  =\mathrm{E}_{\mathcal{T}}\left[\hat{y}_{0}-\mathrm{E}_{\mathcal{T}}\left(\hat{y}_{0}\right)\right]^{2}+\left[\mathrm{E}_{\mathcal{T}}\left(\hat{y}_{0}\right)-f\left(x_{0}\right)\right]^{2} 
  =\operatorname{Var}_{\mathcal{T}}\left(\hat{y}_{0}\right)+\operatorname{Bias}^{2}\left(\hat{y}_{0}\right)
  \end{aligned}$
* In low dimensions and with N = 1000, the nearest neighbor is very close to 0, and so both the bias and variance are small. As the dimension increases, the nearest neighbor tends to stray further from the target point, and both bias and variance are incurred.
* Bias increases as dimension increases with 1-NN, but variance can decrease - function gets very flat far from the test point, so the predictions don’t vary much once all distances are huge. (Page 25). Alternatively, with a different function, the bias is small and the variance takes off - very scenario dependent.

### Statistical Models

* Additive error model for regression $Y=f(X)+\varepsilon$ where random error epsilon has $E(\varepsilon) = 0$, independent of X. This assumptions imply: $f(x)=\mathrm{E}(Y | X=x)$. $P(Y|X)$ depends on X only though f(X) as an assumption - we get the same shape distribution for the errors given any value of X. These model’s approximation that means all unmeasured variables are captured by $\varepsilon$. N realizations from the model and assume errors are independent - $Var(\varepsilon_i) = \sigma^2$
* Generally there will be other unmeasured variables that also contribute to Y, including measurement error. The additive model assumes that we can capture all these departures from a deterministic relationship via the error epsilon.
* Function approximation - The goal is to obtain a useful approximation to f(x) for all x in some region of $\R^p$, given the representations in T. Allows us to use geometrical concepts and probabilistic inference.
* We can write down a loss function without making any assumptions - doesn’t necessarily imply a statistical model. But it is often useful to employ a model instead of pure inference learning from data.
* MLE: $P(G=k|X=x)=p_{k,\theta}(x)$ - vector of functions for G = k. Maximizes log likelihood to pick the parameter.
* Linear basis expansions $f_{\theta}(x)=\sum_{k=1}^{K} h_{k}(x) \theta_{k}$. 
  * $h_k$ are a suitable set of functions or transformations of the input vector x, such as polynomials, trig functions. Or non-linear expansions such as the sigmoid $h_{k}(x)=\frac{1}{1+\exp \left(-x^{T} \beta_{k}\right)}$
  * $\theta$ - set of parameters modified to suit the data at hand, say $\theta = \beta$ for regression. Could estimate through OLS
* We imagine our parameterized function as a surface in p+1 space and we just see noisy realizations of it.
* MLE - for random sample, density $y_{i}, i=1, \ldots, N,\; \operatorname{Pr}_{\theta}(y)$, get log probability $L(\theta)=\sum_{i=1}^{N} \log \operatorname{Pr}_{\theta}\left(y_{i}\right)$
  * For additive model, LS equivalent to MLE using conditional likelihood $\operatorname{Pr}(Y | X, \theta)=N\left(f_{\theta}(X), \sigma^{2}\right)$

### Structured Regression Models

* For arbitrary f, minimizing $\operatorname{RSS}(f)=\sum_{i=1}^{N}\left(y_{i}-f\left(x_{i}\right)\right)^{2}$ has infinite solutions since any f passing through the points is min. We need to restrict the class of function considered. If our prediction fits every training point, under a different training set where for each x there is a slightly different y value, every prediction will have error.
* However If there are multiple observation pairs $x_i, y_{i\ell},  \;\ell= 1,...,N_i$ at each value of $x_i$, the risk is limited - solutions pass through average of $y_{i\ell}$. If sample size is infinite, this is guaranteed, but for finite N, need to reduce set of f’s.
* There are still infinite possible restrictions, so the amiguity still exists. In general the constraints imposed by most learning methods can be described as complexity restrictions of one kind or another. This usually means some kind of regular behavior in small neighborhoods of the input space. How small a neighborhood do we draw within which the function is constant. Very often we impose smoothing conditions - either explicity like setting a linear model or splines, implicity with some nonparametric method.
* Any method that attempts to produce locally varying functions in small isotropic neighborhoods will run into problems in high dimensions—again the curse of dimensionality. And conversely, all methods that overcome the dimensionality problems have an associated—and often implicit or adaptive—metric for measuring neighborhoods, which basically does not allow the neighborhood to be simultaneously small in all directions. If we restrict our attention to a single dimension or a few, we can look closely at distance in those directions.

### Classes of Restricted Estimators

* Roughness penalty and Bayesian methods
  * Penalize RSS with a roughness penalty: $\operatorname{PRSS}(f ; \lambda)=\operatorname{RSS}(f)+\lambda J(f)$. Instead of minimizing RSS through interpolation between all points, we force some smoothing considerations.
  * $J(f)$ will be large for functions that vary too rapidly over small regions of input space. Eg. cubic smoothing spline $\operatorname{PRSS}(f ; \lambda)=\sum_{i=1}^{N}\left(y_{i}-f\left(x_{i}\right)\right)^{2}+\lambda \int\left[f^{\prime \prime}(x)\right]^{2} d x$
  * Penalty function, or regularization methods, express our prior belief that the type of functions we seek exhibit a certain type of smooth behavior, and indeed can usually be cast in a Bayesian framework.
* Kernel Methods and Local Regression
  * Explicitly providing estimates of the regression function or conditional expectation by specifying the nature of the local neighborhood, and of the class of regular functions fitted locally.
  * Kernel function specifies local neighborhood $K_{\lambda}\left(x_{0}, x\right)$. Assigns weights to points x in region around $x_0$. 
  * Eg. local regression minimizing $\operatorname{RSS}\left(f_{\theta}, x_{0}\right)=\sum_{i=1}^{N} K_{\lambda}\left(x_{0}, x_{i}\right)\left(y_{i}-f_{\theta}\left(x_{i}\right)\right)^{2}$
* Basis Functions and Dictionary Methods
  * Includes the familiar linear and polynomial expansions, but more importantly a wide variety of more flexible models. Model f is a linear expansion of basis functions $f_{\theta}(x)=\sum_{m=1}^{M} \theta_{m} h_{m}(x)$. M can be a tuning parameter, choose the # of basis functions.
  * Radial basis functions are symmetric p-dimensional kernels located at particular centroids: $f_{\theta}(x)=\sum_{m=1}^{M} K_{\lambda_{m}}\left(\mu_{m}, x\right) \theta_{m}$

### Model Selection and Bias-Variance Tradoff

* Many flexible methods have a complexity parameter described above, since RSS would produce a perfectly interpolating function. Can have low penalty in low noise situations and come close to fitting data exactly - think of tight fitting of NN for image classification. Such a high signal to noise ratio that danger of overfitting is low.
* $\operatorname{EPE}_{k}\left(x_{0}\right)=\mathrm{E}\left[\left(Y-\hat{f}_{k}\left(x_{0}\right)\right)^{2} | X=x_{0}\right] = \sigma^{2}+\left[\operatorname{Bias}^{2}\left(\hat{f}_{k}\left(x_{0}\right)\right)+\operatorname{Var}_{\mathcal{T}}\left(\hat{f}_{k}\left(x_{0}\right)\right)\right]=\sigma^{2}+\left[f\left(x_{0}\right)-\frac{1}{k} \sum_{\ell=1}^{k} f\left(x_{(\ell)}\right)\right]^{2}+\frac{\sigma^{2}}{k}$ for $ \mathcal{T}$= training data. $Y|x_0$ is independent of $\mathcal{T}$. $\sigma^2_{x_0}$ is the variance of Y arouund x0.
* Note $\hat{f}(X_0)$ is a RV, it depends on the training data - this leads directly to our understanding of bias and variance 
* The first term $\sigma^2$ is the irreducible error. The second and third terms are under our control, and make up the mean squared error, broken down into a bias component and a variance component. 
* Bias: $\left[\mathrm{E}_{\mathcal{T}}\left(\dot{f}_{k}\left(x_{0}\right)\right)-f\left(x_{0}\right)\right]^{2}$ square difference between true mean and expected value of the estimate.
* When we calculated EPE, consider what is random. Treating $x_i's$ as fixed, the randomness if from the training Y’s that are associated with the x’s in the training set. Then we can say $\mathrm{E}_{Y | X}\left[\left(Y-\hat{f}_{k}\left(x_{0}\right)\right)^{2} | X=x_{0}\right]=\sigma^{2}+\operatorname{Bias}^{2}\left(\hat{f}_{k}\left(x_{0}\right)\right)+\operatorname{Var}_{\mathcal{T}}\left(\hat{f}_{k}\left(x_{0}\right)\right)=\sigma^{2}+\left[f\left(x_{0}\right)-\frac{1}{k} \sum_{\ell=1}^{k} f\left(x_{(\ell)}\right)\right]^{2}+\frac{\sigma^{2}}{k}$ As k gets big in KNN, the variance term declines but the bias term increases since we are averaging over more f(x) values.
* The variance term is simply the variance of an average here, and decreases as the inverse of k.
* Often prediction error (EPE) is a vehicle for getting at MSE, which tells how well we are estimating f(x), our real interest.
* Performance of OLS vs KNN in Bias Variance Terms
  * OLS: $EPE(X_0) \approx \sigma^2 \times \frac{P}{N}\sigma^2 + bias^2_{OLS}(x_0)$
    * How did we get P/N: $\hat{\beta} = (X^TX)^{-1}X^Ty$ and $\hat{f} = X\hat{\beta}$. Say $Y \sim f(x) + \epsilon$ for $\epsilon \sim (0,\sigma^2)$. Treat X’s in training sample as fixed, condition on those X’s, then $\hat{f} = X\hat{\beta}=Ay$. $Cov(Y) = \sigma^2I_n$ so $Cov(\hat{f}) = \sigma = A\sigma^2$. What is the avg variance of $\hat{f}_i$? $Var(\hat{f}_i) = tr(A)\sigma^2/n$ where $tr(A) = p =$ # of parameters. So this is close to P/N sigma squared
  * KNN: $EPE(X_0) \ge \sigma^2 + \sigma^2 + bias^2_{1-NN}(x_0)$
    * Sigma squared - both irreducible error and the error from the closest neighbor. 
  * Concluding $\sigma^2 + P\sigma^2/N \approx \sigma^2 < 2\sigma^2$ - OLS better on variance. f(x) determines the bias, but for large N KNN could be optimal.

## Chapter 3: Linear Methods for Regression

* Function in nature are generally not linear, so we should always think of our model as an approximation. 
* RSE is divided by n-p-1 - we have already done some fitting, so we have reduced our degrees of freedom by p. This ensures RSE is unbiased estimate.
* Hat matrix is conditonal on X - assuming that x is not random. So when we do $Var(\hat{\beta}) = (X^TX)^{-1}\sigma^2$, $\hat{\beta} = My$, $Var(\hat{\beta}) = MVar(y)M^T = \sigma^2MM^T = \sigma^2(X^TX)^{-1}$ since $Var(y) = \sigma^2I_n$ and $M=(X^TX)^{-1}X^T$

### Least Squares

* Basic model: $f(X)=\beta_{0}+\sum_{j=1}^{p} X_{j} \beta_{j}$  - the model is linear in the parameters, but X can be a number of different forms
* Minimize RSS: $\operatorname{RSS}(\beta)=\sum_{i=1}^{N}\left(y_{i}-f\left(x_{i}\right)\right)^{2} = (\mathbf{y}-\mathbf{X} \beta)^{T}(\mathbf{y}-\mathbf{X} \beta)$ and when X has full rank, $\hat{\beta}=\left(\mathbf{X}^{T} \mathbf{X}\right)^{-1} \mathbf{X}^{T} \mathbf{y}$
* The outcome vector y is orthogonally projected onto the hyperplane spanned by the input vectors x1 and x2. The projection $\hat{y}$ represents the vector of the least squares predictions: $\hat{\mathbf{y}}=\mathbf{X} \hat{\beta}=\mathbf{X}\left(\mathbf{X}^{T} \mathbf{X}\right)^{-1} \mathbf{X}^{T} \mathbf{y} = Hy$
* The p predictors span a p-dimensional subspace of N space. Want to find an approximation of y that lies in that subspace - a linear combination of the columns of X. Necessarily the closest y is orthogonal to the subspace - $\hat{y} \perp $ subspace. Essentially $(y-\hat{y})\perp x_j \forall j$ - e is perp to all x vectors that form a basis for the subspace. Could choose many betas that project onto the subspace - but the one with the smallest e is the least square solution.
* If two inputs perfectly correlated, X won’t be full rank. The least square projection no longer works, but $\hat{y}$ is still the projection of y onto C(X), there is just more than one way to express the projection in terms of the X columns.
* To do more with this model, we need to assume the conditional expectation of Y is linear in $X_1,...,X_p$ and deviations of Y around its expectation are additive and Gaussian - $\varepsilon \sim N\left(0, \sigma^{2}\right)$. Then $\hat{\beta} \sim N\left(\beta,\left(\mathbf{X}^{T} \mathbf{X}\right)^{-1} \sigma^{2}\right)$
* F-Statisic: $F=\frac{\left(\mathrm{RSS}_{0}-\mathrm{RSS}_{1}\right) /\left(p_{1}-p_{0}\right)}{\mathrm{RSS}_{1} /\left(N-p_{1}-1\right)}$ measures the change in residual sum-of-squares per additional parameter in the bigger model, and it is normalized by an estimate of $\sigma^2$.
  * More general from F test: say $H_0:\beta_{1,...,p-q}\neq 0$ and last q = 0. Then $(RSS_0 - RSS /q)/(RSS/n-p)\sim F_{n-p},q$ for $RSS_0$ the RSS for our subset model forcing the last q to 0. When q = 1, $F_{n-p,1} = t^2_{n-p}$ - equivalence between t and F tests. Then the p values can be interpreted as the effect you get leaving that variable out of the model and keeping the others.
  * Cannot simply remove the insignificant variables - if they are linearly dependent, then removing one might make the other significant. 
* Properties of OLS 
  * OLS is **equivariant** under non-singular linear transformations of X. If we make a transformation and solve we can undo the transformation to recover the solution for the original system. If $\hat{\beta}$ is OLS solution for X then $\beta^* = A^{-1}\hat{\beta}$ is OLS solution for $X^* = XA$ for A p x p nonsingular.
    * Many models are not equivariant - PCA is not equivariant. $z_1 = Xa_1$ - if we standardized the cols of X we get a solution but not recoverable if we didn’t standardize the cols of X. Lasso and ridge are also not equivariant.
  * Given $$\mathbf{z}_{p}=\mathbf{x}_{p}-\mathbf{X}_{(p)} \gamma$$ for $\mathbf{X}_{(p)}$ submatrix of X exluding last columns for any gamma. Then OLS coef for $x_p$ is the same as for $z_p$. This follows from equivariance. We have $X = (X_{(p)},x_p)$, $\tilde{X} = (X_{(p)},z_p)$. Regression on X and $\tilde{X}$, then $x_p, z_p$ fill the same role in the regression.
    * Now let $\gamma$ be the OLS coef of $x_p$ on $X_{(p)}$, then $z_p$ is the residual obtained by adjusting $x_p$ for other variables in the model. $z_p$ then orthogonal to $X_{(p)}$. 
    * Why do we care? The coef of a variable picks up what is left off after removing the effect of others. The variance of a coefficient depends on the size of the norm of $z_p$. If the residual is small after adjusting for the other variables, the norm gets small and the variance of the coef blows up - highly uncertain when not much unexplained signal left in the residuals. This is true for any of the variates. The more correlated stuff you throw into your model, the less interpretable your model will be.

##### Assumptions for Inference

* Assume linearity, normality, constant variance and independent of errors with X’s.
* Often the last is ignored - given some model $y_{i}=\beta_{0}+\sum_{j} x_{i j} \beta_{j}+\left(\sum_{k} x_{i k}^{*} \theta_{k}+\varepsilon_{i}\right)$ we often just lump in the last ones with the error.
* Also the x’s are considered fixed, but just an assumption of convenience - most of the time x’s are random. 
* **Bootstrap can help here**
* We assume additivity, but predictors usually change together - correlation in the data can confound interpretation. Can often end up with negative effect on one predictor when two correlated predictors are both positively correlated with the response. Can render a model uninterpretable

##### Model Performance

* Measure of estimator: $$\operatorname{Mse}[\hat{\mathbf{f}}(x)]=\mathrm{E}[\hat{\mathbf{f}}(x)-\mathbf{f}(x)]^{2} = \operatorname{Var}[\hat{\mathbf{f}}(x)]+[\mathrm{E} \hat{\mathbf{f}}(x)-\mathbf{f}(x)]^{2}$$
* If linear model is correct, Gauss-Markov theorem applies and the LS prediction is unbiased and has the lowest variance among all unbiased estimators that are linear functions of y. 
* Application: Time Series
  * Fitting AR(k) model - k lags as predictors.
  * We regress over many correlated variables, the t-stats are not significant, but prediction is our goal, not inference.

##### Gauss Markov Theorem

* Assumptions: linear model is correct, unbiased estimate, linear estimate. These are hard assumptions to live up to in practice. Always we can create a biased estimate with smaller MSE, James-Stein theorem says we can always improve an estimator’s MSE through regularization / shrinkage.
* The least squares estimates of the parameters $\beta$ have the smallest variance among all linear unbiased estimates.
* The Gauss-Markov theorem implies that the least squares estimator has the smallest mean squared error of all linear estimators with no bias. However, there may well exist a biased estimator with smaller mean squared error.

##### Multiple Regression

* From simple regression: $Y=X \beta+\varepsilon$, $\begin{aligned}
  \hat{\beta}=\frac{\langle\mathbf{x}, \mathbf{y}\rangle}{\langle\mathbf{x}, \mathbf{x}\rangle},\;
  \mathbf{r}=\mathbf{y}-\mathbf{x} \hat{\beta}
  \end{aligned}$ for residuals r. 
* When the inputs of X are orthogonal, they have no effect on each other’s parameter estimates in the model. Very rare in observational data, so need to orthogonalize them.
* Saying formally, $X_1, X_2$ mutually orthogonal, then $X_1^T(y - X\hat{\beta}) = X_1^T(y - X_1\hat{\beta}_1)=0$ and same for $\beta_2$. You can do the multiple regression by doing the univaritate regression with orthogonal columns.
* Gram Schmidt for multiple regression
  * Initialize $z_0 = x_0 = 1$. 
  * For $j \in 1,...,p$, regress $x_j$ onto $z_0,...,z_j-1$ to produce coefficients $\hat{\gamma}_{lj}=\left\langle\mathbf{z}_{\ell}, \mathbf{x}_{j}\right\rangle /\left\langle\mathbf{z}_{\ell}, \mathbf{z}_{\ell}\right\rangle$ for l in 0 to j-1 and residual vector $z_j = \mathbf{x}_{j}-\sum_{k=0}^{j-1} \hat{\gamma}_{k j} \mathbf{z}_{k}$. This procedure is equivalent to $\mathbf{X}=\mathbf{Z} \mathbf{\Gamma} = \mathbf{Z D}^{-1} \mathbf{D} \mathbf{\Gamma}=\mathbf{Q} \mathbf{R}$
  * Regress y on the residual $z_p$ to get estimate $\hat{\beta}_{p}$
* The algorithm produces $\hat{\beta}_{p}=\frac{\left\langle\mathbf{z}_{p}, \mathbf{y}\right\rangle}{\left\langle\mathbf{z}_{p}, \mathbf{z}_{p}\right\rangle}$ Since the zj are all orthogonal, they form a basis for the column space of X, and hence the least squares projection onto this subspace is $\hat{y}$. 
* The multiple regression coefficient $\beta_j$ represents the additional contribution of xj on y, after xj has been adjusted for x0, x1,..., xj−1, xj+1,..., xp. If xp is highly correlated with some of the other xk’s, the residual vector zp will be close to zero, and from (3.28) the coefficient $\hat{\beta}_p$ will be very unstable. See via this formula $\operatorname{Var}\left(\hat{\beta}_{p}\right)=\frac{\sigma^{2}}{\left\langle\mathbf{z}_{p}, \mathbf{z}_{p}\right\rangle}=\frac{\sigma^{2}}{\left\|\mathbf{z}_{p}\right\|^{2}}$

### Regression via QR Decomposition

* Think of relation of Q and R. The first column of x is the first column of Q times a single number. Vector $q_1 = \frac{x_1}{||x_1||_2}$, the top element of R is $||x_1||_2$ to bring us back to X.  The first p columns of X for a basis for the columns of X. 
* The second column of X is a linear combination of the first two columns of Q - hence Q’s second column has the elements of the first column substracted off.
* If X is not full rank, then first r columns of R look the same, but then we get a deficient triangle and get 0’s towards the bottom of the triangle. Don’t need the later columns in the linear combinations that produce X. This assumes that the columns are in a certain order that allow us to do this - we can shuffle the columns and pull the independent columns of X forward.
* With an L2 norm, can insert an orthogonal matrix inside of it and it won’t change the norm: $||a||_2^2 = ||Qa||_2^2 \implies a^TQ^TQa = a^Ta$
* Using this fact, for the full rank X, $\begin{aligned}
  \|y-X \beta\|^{2} &=\left\|Q^{T} y-R \beta\right\|^{2}
  =\left\|Q_{1}^{T} y-R_{1} \beta\right\|^{2}+\left\|Q_{2}^{T} y\right\|^{2}
  \end{aligned}$. (Steps to get here: $X=QR \implies y=QR\beta \implies Q^Ty=Q^TQR\beta \implies y=R\beta$). Then $\hat{\beta}=R_{1}^{-1} Q_{1}^{T} y$ from the first term leaving $\mathbf{R S S}(\hat{\beta})=\left\|Q_{2}^{T} y\right\|^{2}$
  * When you have a euclidean norm, you can always multiply inside on the left by an ortho matrix. Why? $||Z||^2  = Z^TZ \implies (QZ)^TQZ = Z^TZ$
  * $R_1$ above is p x p nonsingular. The inverse of triangular matrix is trivial.
* $e=Q^{T} y$ - coordinates of y on columns of Q - ie. in an orthonormal basis. $H = Q_1Q_1^T$. For rank of X k < p, we solve $Q_{1}^{T} y=R_{11} \beta_{1}+R_{12} \beta_{2}$ where $Q_1$ has r columns - this has infinite solutions. 
* $\hat{y}$ is a projection of y into the columns space of X, so $\hat{y} = Q_1Q_1^ty = Hy = X(X^TX)^{-1}X^T$ for projection matrix hat H. Clearly the orthornormal version of H is more convenient. When rank X is deficient, this is especially useful. $Q_1$ is the first r columns of Q, and all columns beyond that are going to be zero when multiplied by R. Then $Q_{1}^{T} y=R_{11} \beta_{1}+R_{12} \beta_{2}$ has infinite solutions, need to set $\beta_2=0$ and solve for $\beta_1$, but this is an arbitrary choice. We could have chosen other solutions. We can do something less arbitrary - finding the $\beta$ with the smallest norm (see Strang for min solution).
* Despite the non-uniqueness of the solution, the fit is well defined - we still are projecting onto the same subspace $Q_1Q_1^Ty$.

##### Distributional Aspects

* $\varepsilon \sim (0, \sigma^2)$ iid, though we can add normality. Given the X’s (fixed), $Cov(\hat{\beta}) = (X^TX)^{-1}\sigma^2 = (R^TR)^{-1}\sigma^2$
* Adding the normal assumption for errors, we get the beta distribution $\hat{\beta} \sim N(\beta, (X^TX)^{-1}\sigma^2)$
* The effects e also normal $N(R\beta, \sigma^2 I)$ (since y was distributed $Q\beta$). Can break e into e1 and e2, wheree1 has mean $R_1\beta$ and e2 have mean zero since remaining R is zero. Then $||e_2||^2 \sim \sigma^2\chi^2_{N-p}$
* Note $\sigma^2 = RSS / (n-p)$, denom to make unbiased. e1 distributed with a non zero mean, but under H0, $\beta =0$, so e1 also chi distributed. e1 and e2 independent since uncorrelated Normals are independent. Then $$\frac{\left\|e_{1}\right\|^{2}}{p} / \frac{\left\|e_{2}\right\|^{2}}{N-p} \sim F_{p, N-p}$$ which we can use to test all the coefficients are simultaneously 0.

### Subset Selection

* Prediction accuracy: the least squares estimates often have low bias but large variance, may be improved by taking on some bias via shrinkage and selection. Interpretation: With a large number of predictors, we often would like to determine a smaller subset that exhibit the strongest effects.
* Best subset regression finds for each $k \in \{0, 1, 2,...,p\}$ the subset of size k that gives smallest residual sum of squares
* Forward stepwise selection starts with the intercept, and then sequentially adds into the model the predictor that most improves the fit. Computational: for large p we cannot compute the best subset sequence, but we can always compute the forward stepwise sequence. Statistical: forward stepwise is a more constrained search, and will have lower variance, but perhaps more bias.
* Backward-stepwise selection starts with the full model, and sequentially deletes the predictor that has the least impact on the fit. The candidate for dropping is the variable with the smallest Z-score. Backward selection can only be used when N>p, while forward stepwise can always be used.
* Forward-stagewise regression (FS) is even more constrained than forward stepwise regression. At each step the algorithm identifies the variable most correlated with the current residual. It then computes the simple linear regression coefficient of the residual on this chosen variable, and then adds it to the current coefficient for that variable. This is continued till none of the variables have correlation with the residuals (the least square fit when N > p). Unlike forward-stepwise regression, none of the other variables are adjusted when a term is added to the model. As a consequence, forward stagewise can take many more than p steps to reach the least squares fit - slow fitting.

### Shrinkage

##### Ridge Regression

* $\hat{\beta}^{\text {ridge }}=\underset{\beta}{\operatorname{argmin}}\left\{\sum_{i=1}^{N}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} x_{i j} \beta_{j}\right)^{2}+\lambda \sum_{j=1}^{p} \beta_{j}^{2}\right\}$ or equivalently $\hat{\beta}^{\text {ridge }}=\underset{\beta}{\operatorname{argmin}} \sum_{i=1}^{N}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} x_{i j} \beta_{j}\right)^{2}$  st $\sum_{j=1}^{p} \beta_{j}^{2} \leq t$
* Here $\lambda \geq 0$ is a complexity parameter that controls the amount of shrinkage: the larger the value of $\lambda$, the greater the amount of shrinkage. 
* Ridge solutions dependent on scale - should standardize inputs! Note $\beta_0$ is not in penalty term, otherwise adding a constant to Y would effect results non linearly.
* In matrix notation, take centered inputs $x_{i j}-\bar{x}_{j}$ less the constant. Using centered X, $\operatorname{RSS}(\lambda)=(\mathbf{y}-\mathbf{X} \beta)^{T}(\mathbf{y}-\mathbf{X} \beta)+\lambda \beta^{T} \beta$ and the solutions are $\hat{\beta}^{\text {ridge }}=\left(\mathbf{X}^{T} \mathbf{X}+\lambda \mathbf{I}\right)^{-1} \mathbf{X}^{T} \mathbf{y}$
* Note by adding a constant to $X^TX$, it is always non-singular and is always invertible. In the case of orthonormal inputs, ridge edstimates are just scaled LS estimates $\hat{\beta}^{\text {ridge }}=\hat{\beta} /(1+\lambda)$
* Effective degrees of freedom: $\begin{aligned}
  \mathrm{df}(\lambda) &=\operatorname{tr}\left[\mathbf{X}\left(\mathbf{X}^{T} \mathbf{X}+\lambda \mathbf{I}\right)^{-1} \mathbf{X}^{T}\right] 
  =\operatorname{tr}\left(\mathbf{H}_{\lambda}\right) 
  =\sum_{j=1}^{p} \frac{d_{j}^{2}}{d_{j}^{2}+\lambda}
  \end{aligned}$ for singular values d.

##### Lasso

* $\hat{\beta}^{\text {lasso }}=\underset{\beta}{\operatorname{argmin}}\left\{\frac{1}{2} \sum_{i=1}^{N}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} x_{i j} \beta_{j}\right)^{2}+\lambda \sum_{j=1}^{p}\left|\beta_{j}\right|\right\}$ or equivalently $\hat{\beta}^{\text {lasso }}=\underset{\beta}{\operatorname{argmin}} \sum_{i=1}^{N}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} x_{i j} \beta_{j}\right)^{2}$ st $\sum_{j=1}^{p}\left|\beta_{j}\right| \leq t$
* Small t -> coefficients shrink to 0, form of subset selection. Larger t, end up with LS estimates.

### Regression Restriction Comparisons

* Ridge regression does a proportional shrinkage. Lasso translates each coefficient by a constant factor $\lambda$, truncating at zero. This is called “soft thresholding”
* Both methods find the first point where the elliptical contours hit the constraint region. Unlike the disk, the diamond has corners; if the solution occurs at a corner, then it has one parameter $\beta_j$ equal to zero.
* Consider penalty $\lambda \sum_{j=1}^{p}\left|\beta_{j}\right|^{q}$; each $\left|\beta_{j}\right|^{q}$ can be a log-prior density for $\beta_j$. The lasso, ridge regression and best subset selection are Bayes estimates with different priors.
* This generic penalty constructs different constraint contours, but little values for q > 2 and between 0 and 1 becomes non convex.
* Elastic-net Penalty - compromise between ridge and lasso: $\lambda \sum_{j=1}^{p}\left(\alpha \beta_{j}^{2}+(1-\alpha)\left|\beta_{j}\right|\right)$. Has constraint contour like lasso with slightly rounded edges.