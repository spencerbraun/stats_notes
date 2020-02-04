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
* Best / all subset regression finds for each $k \in \{0, 1, 2,...,p\}$ the subset of size k that gives smallest residual sum of squares. 
  * The null model has the full variance of the dataset - just predicted by the mean. The number of models increases around the middle of subset size k. The smaller k’s have clear winners, but by middle k’s the models are bunch up in performance. Good to note since the “ideal” subset is not particularly differentiated from other model choices.
* Forward stepwise selection starts with the intercept, and then sequentially adds into the model the predictor that most improves the fit. Computational: for large p we cannot compute the best subset sequence, but we can always compute the forward stepwise sequence. Statistical: forward stepwise is a more constrained search, and will have lower variance, but perhaps more bias.
* Backward-stepwise selection starts with the full model, and sequentially deletes the predictor that has the least impact on the fit. The candidate for dropping is the variable with the smallest Z-score. Backward selection can only be used when N>p, while forward stepwise can always be used.
* Could view subset size a tuning parameter for all of these methods. For each size we have the “best” model, then pick the right size.
* Generally need to perform model assessment, could use validation and test sets. Validation use for choosing k, then test set for reporting the performance of the chosen model, since otherwise using one test/validation set includes some bias in selection. With insufficient data, we turn to CV.
* Forward-stagewise regression (FS) is even more constrained than forward stepwise regression. At each step the algorithm identifies the variable most correlated with the current residual. It then computes the simple linear regression coefficient of the residual on this chosen variable, and then adds it to the current coefficient for that variable. This is continued till none of the variables have correlation with the residuals (the least square fit when N > p). Unlike forward-stepwise regression, none of the other variables are adjusted when a term is added to the model. As a consequence, forward stagewise can take many more than p steps to reach the least squares fit - slow fitting.

### Model Performance - CV + Bootstrap

* K-fold CV - divide the data into K equal parts (5,10), for each k fit the model with parameter $\lambda$ to the other K-1 parts, computing the error in predicting the kth part. The total CV error: $C V(\lambda)=(1 / K) \sum_{k=1}^{K} E_{k}(\lambda)$
  * For best subset selection, say, these subsets will be different. The subset of 3 fit to the k in 1,2,3 could be different from the one fit on 2,3,4. But the idea is we are finding the best subset number - not the actual make-up of the subset. The best model within a K may not be too far from another model.
  * Once we have selected the subset size, we go back to the full data set, fit the full sequence of models and choose the model for the tuned subset size. The actual model comes from the full data
  * We tend to prefer 10 to 5 for K for this reason - fitting to 9/10 of the data is closer to fitting to your full training set. 
  * Could go all the way with LOOCV, but computationally difficult (at least for best subset, though not necessarily for something like ridge). LOOCV could also have too high variance compared to K-fold, but the best method depends on the goal and situation.
* Boostrap Approach
  * Can use bootstrap for prediction error estimation. We get a full size training sample since we sample with replacement. 
  * For each bootstrap sample, estimate errors using only observations excluded from the boostrap sample.
  * Pretends the training sample is the population distribution, draws a sample, and the population is the full training sample.
  * Can be much more computationally expensive, but often can have slightly better results
* CV Issues
  * Example: 1) Have 5000 predictors, find 200 predictors having the largest correlation with class labels. 2) Then carry about nearest centroid classification using those 200 predictors
    * Wrong approach is to apply CV to step 2 - need to apply CV to 1 and 2. 
  * Need a CV and test set - if we run CV and get a CV error, it will be slightly biased downwards. If our model has many tuning parameters, we will have to CV multiple times - we have overfit the data for fitting the models and for CV. We need to report our methods error on the test set, a dataset reserved for final evaluation, and of course this could still overreport performance since future datasets may not be the same as today's.
  * If we have little data, a separate test set may be expensive. Could instead use nested CV, using say K-fold, select lambda and train model using selected lambda on the 9/10, test on the 1/10. Grows exponentially in computation needed
  * Example: Does CV really work?
    * Null situation, N = 20 p = 500, where labels independent of data - expect 50% error rate since no signal
    * Take a stump - univaritate classifier that is a single split minimizes the misclassification error. Among all features, pick the feature and best split point to minimize error by a single division.
    * Fit to entire training set, by chance we can find a split that fits data well. In 5-fold CV expect good separation on this feature too. Does this mean CV is useless?
    * With very high number of noise variables, these things will happen by chance. Running this process many times, on average get 50% misclassification but the error rate has high variance - from 0.2 to 0.8. Have to realize with small data can have huge variance, but CV does not produce biased estimate.
    * In practice, just have one number, the CV error, not the full variance of the CV error. In practice, often use variance from the 10 folds but this tends to be too optimistic. Somewhat of an open question about how to improve on that. The ten folds are independent but the estimates are correlated since they rely on overlapping observations - we have bias in our estimates.
* 1 SE Rule - find model with the CV minimum, then take the simplest model within the 1 SE bounds of this best model.

### Shrinkage

##### Ridge Regression

* $\hat{\beta}^{\text {ridge }}=\underset{\beta}{\operatorname{argmin}}\left\{\sum_{i=1}^{N}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} x_{i j} \beta_{j}\right)^{2}+\lambda \sum_{j=1}^{p} \beta_{j}^{2}\right\}$ or equivalently $\hat{\beta}^{\text {ridge }}=\underset{\beta}{\operatorname{argmin}} \sum_{i=1}^{N}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} x_{i j} \beta_{j}\right)^{2}$  st $\sum_{j=1}^{p} \beta_{j}^{2} \leq t$
* Here $\lambda \geq 0$ is a complexity parameter that controls the amount of shrinkage: the larger the value of $\lambda$, the greater the amount of shrinkage. 
* Ridge solutions dependent on scale - should standardize inputs! Note $\beta_0$ is not in penalty term, otherwise adding a constant to Y would affect results non linearly. Easiest way is to center variables - remove mean from X and y.
* In matrix notation, take centered inputs $x_{i j}-\bar{x}_{j}$ less the constant. Using centered X, $\operatorname{RSS}(\lambda)=(\mathbf{y}-\mathbf{X} \beta)^{T}(\mathbf{y}-\mathbf{X} \beta)+\lambda \beta^{T} \beta$ and the solutions are $\hat{\beta}^{\text {ridge }}=\left(\mathbf{X}^{T} \mathbf{X}+\lambda \mathbf{I}\right)^{-1} \mathbf{X}^{T} \mathbf{y}$
  * In this form we can simply take the gradient, $-2X^T(y - X\beta) + 2\lambda \beta=0$ and solve.
* There is always a value of $\lambda$ that reduces the MSE more than the OLS estimator.
  * $y \sim X\beta + \varepsilon$, then some value of lambda > 0 minimizes $E||\beta -\hat{\beta}_\lambda||_2^2$
* Note by adding a constant to $X^TX$, it is always non-singular and is always invertible. In the case of orthonormal inputs, ridge edstimates are just scaled LS estimates $\hat{\beta}^{\text {ridge }}=\hat{\beta} /(1+\lambda)$
* Effective degrees of freedom: $\begin{aligned}
  \mathrm{df}(\lambda) &=\operatorname{tr}\left[\mathbf{X}\left(\mathbf{X}^{T} \mathbf{X}+\lambda \mathbf{I}\right)^{-1} \mathbf{X}^{T}\right] 
  =\operatorname{tr}\left(\mathbf{H}_{\lambda}\right) 
  =\sum_{j=1}^{p} \frac{d_{j}^{2}}{d_{j}^{2}+\lambda}
  \end{aligned}$ for singular values d.
  * When lambda is 0, the df is the number of predictors in the model. When lambda infinity, all the coefficients are 0, so df=0. 
  * In the linear model, we get $\hat{y} = Hy$ where H is a projection operator. The trace of H gives you the number of parameters in the model. Use similar idea for df here - $df(\lambda) = tr(H_\lambda)$

##### Lasso

* $\hat{\beta}^{\text {lasso }}=\underset{\beta}{\operatorname{argmin}}\left\{\frac{1}{2} \sum_{i=1}^{N}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} x_{i j} \beta_{j}\right)^{2}+\lambda \sum_{j=1}^{p}\left|\beta_{j}\right|\right\}$ or equivalently $\hat{\beta}^{\text {lasso }}=\underset{\beta}{\operatorname{argmin}} \sum_{i=1}^{N}\left(y_{i}-\beta_{0}-\sum_{j=1}^{p} x_{i j} \beta_{j}\right)^{2}$ st $\sum_{j=1}^{p}\left|\beta_{j}\right| \leq t$
* Small t -> coefficients shrink to 0, form of subset selection. Larger t, end up with LS estimates.
* Acts in a nonlinear manner on the outcome unlike ridge. It sets some coefficients to 0 for large enough penalty - acts in between subset selection and ridge regularization. Can think of it as a convex relaxation of the subset size constraint.
* Can think of ridge as the OLS line with a smaller slope, The lasso is the OLS line with a parallel shift downwards, setting negative values to zero - kinked.
* For ridge and lasso imporant to standardize predictors to mean zero and SD 1. If all predictors are in the same units, could just center but not scale.
* Degrees of freedom for lasso is harder. Can use the more general definition $df(\hat{y}) = \sum_i Cov(y_i, \hat{y}_i / \sigma^2)$ - average covariance over a point in the data and its fitted value
  * Covariance on repeated training samples. If a method overfits, prediction closer to y in data and covariance is high. If we use the null model then covariance will be quite small. 
  * For ridge regression, have y and $\hat{y}_\lambda = H_\lambda y$, then want the $Cov(\hat{y}_\lambda,y) = Cov(H_\lambda y,y)=\sigma^2 H_\lambda $ for $y \sim(f, \sigma^2I)$. So the sum of the covariances is $Tr(\sigma^2H_\lambda)$, which is what we found before, so this is a generalization of the ridge dfs
  * Take size of the active set (# of parameters fit in lasso / size of subset) $\hat{k}_\lambda$ is unbiased in the df - $E(\hat{k}_\lambda) = df(\lambda)$
  * This is not true for best subset, searched over a state space. In Lasso, we not only get a subset but we also get shrinkage in the  coefficients. This is enough to bring the degrees of freedom down to k.
* Notice in the lasso coefficient graph - coefficients start growing at different times. In ridge, all coefficients started growing once we alter the shrinkage factor. 
* Homotopy path of the lasso is piecewise linear. In between where active set changes get a piecewise linear function, notice knots on graph. When the set changes, all of the coefficients change across variables in the model. This means you can compute the whole lasso path in the same time as the least squares fit - lars algorithm. We know the order in which to add predictors.

### Regression Restriction Comparisons

* Ridge regression does a proportional shrinkage. Lasso translates each coefficient by a constant factor $\lambda$, truncating at zero. This is called “soft thresholding”
* Both ridge and lasso, centering allows us to forget the intercept.
* Both methods find the first point where the elliptical contours hit the constraint region. Unlike the disk, the diamond has corners; if the solution occurs at a corner, then it has one parameter $\beta_j$ equal to zero.
* Family of Shrinkage Estimators: Consider penalty $\lambda \sum_{j=1}^{p}\left|\beta_{j}\right|^{q}$; each $\left|\beta_{j}\right|^{q}$ can be a log-prior density for $\beta_j$. The lasso, ridge regression and best subset selection are Bayes estimates with different priors.
* This generic penalty constructs different constraint contours, but little values for q > 2 and between 0 and 1 becomes non convex.
* Elastic-net Penalty - compromise between ridge and lasso: $\lambda \sum_{j=1}^{p}\left(\alpha \beta_{j}^{2}+(1-\alpha)\left|\beta_{j}\right|\right)$. Has constraint contour like lasso with slightly rounded edges. Of course you get another parameter alpha, but often don’t need a fine search over it like you do for lambda.
* If you do ridge with identical predictors, they end up sharing the coefficient equally. Lasso does not - two variables identical in the model, could split a coefficient any number of ways, ambivalent about the linear combination. If you have predictors in groups, and want the group to be selected, elastic-net removes some of the arbitrariness of lasso selection among correlated groups.

### Derived Input Direction Methods

##### PC Regression

* Principal components depend on the scaling of the inputs, so typically we first standardize them.
* For X data matrix and $\tilde{X}$ the data matrix with centered columns. The largest PC direction v max’s $\hat{Var}(Xv) = \frac{1}{N}v^T\tilde{X}^T\tilde{X} v$ subject to $||v||_2 =1$. Then $z_1 = \tilde{X}v_1$ is the largest eigenvector of the coviance matrix. $z_2 = \tilde{X}v$ where we want the variance between z1 and z2 to be zero. So take $\frac{1}{N}v^T\tilde{X}^T\tilde{X} v = \frac{d^2}{n_1}v_1^Tv = 0$. Do the full eigendecomposition and get the PCs. All of the PCs defined by eigenvector equations $\frac{1}{N} \tilde{\mathbf{X}}^{T} \tilde{\mathbf{X}} \mathbf{v}_{j}=d_{j}^{2} \mathbf{v}_{j}, \quad j=1, \ldots, p$. Also note that the SVD of the centered X provides the PCs of X.
* PCR - regress y on z1,...zj. Since all zj are orthogonal, it is just a sum of univariate regressions.
* If M = p, we would just get back the usual least squares estimates; for M<p we get a reduced regression.
* Implicit assumption that the response will change more in the direction that the x’s change most.
* We see that principal components regression is very similar to ridge regression: both operate via the principal components of the input matrix. Ridge regression shrinks the coefficients of the principal components, shrinking more depending on the size of the corresponding eigenvalue; principal components regression discards the p − M smallest eigenvalue components.
  * Ridge: for $\tilde{X} = UDV^T,\; Z = \tilde{X}V = UD$ - the left singular vectors scaled by the singular values gives you Z. We are regressing on Z, so scaling by D does not matter, so can just use U, an orthonormal basis. So $\hat{y}_\lambda = \sum_{j=1}^p u_j \frac{d_j^2}{d_j^2 + \lambda}\langle u_j,y\rangle$ - gives more weight to leading PCs but then shrinkage increases. Bigger lambda, shrinkage squeezes more.
  * For PCR: $\hat{y} = \sum_{j=1}^k u_j \langle u_j, y \rangle$ - gives weight 1 to the first k components, weight 0 to the rest. As we shrink k, number of PCs included, more will drop out.

##### Partial Least Squares

* Also depends on the scaling of the inputs, so typically we first standardize them to mean 0, var 1
* Akin to PCR but takes the response into account instead of using an unsupervised preprocessing. Fit univariate coefficients then construct z = sum of coefs times x_i. This is the first z, z1. Then regression of y on z1, to get coef beta1. Then orthogonalize y and x’s wrt z1 - nothing left of z1 in any portion of the data. Then repeat the process.
* As with principal-component regression, if we were to construct all M = p directions, we would get back a solution equivalent to the usual least squares estimates; using M<p directions produces a reduced regression.
* It can be shown that partial least squares seeks directions that have high variance and have high correlation with the response, but typically the variance dominates making it close to PCR. 
* If the input matrix X is orthogonal, then partial least squares finds the least squares estimates after m = 1 steps.

### Comparison of Selection and Shrinkage

* For minimizing prediction error, ridge regression is generally preferable to variable subset selection, principal components regression and partial least squares.
* PLS, PCR and ridge regression tend to behave similarly. Ridge regression may be preferred because it shrinks smoothly, rather than in discrete steps. Lasso falls somewhere between ridge regression and best subset regression, and enjoys some of the properties of each.

## Chapter 4: Linear Methods for Classification

* Logit transformation: $\log \frac{\operatorname{Pr}(G=1 | X=x)}{\operatorname{Pr}(G=2 | X=x)}=\beta_{0}+\beta^{T} x$

### Linear Regression of Indicator Matrix

*  $\hat{\mathbf{Y}}=\mathbf{X}\left(\mathbf{X}^{T} \mathbf{X}\right)^{-1} \mathbf{X}^{T} \mathbf{Y}$. Could view it as estimate of conditional expectation, but can exceed the domain of probability densities.
*  Response variable g that takes values in an unordered set of size k. We encode the response in an indicator matrix Y n x k - one hot encoding. We regress this matrix of indicators on X. For the kth column get a linear model $f_{k}(x)=\beta_{k 0}+\beta_{k}^{T} x \quad k=1, \ldots, K$
*  The decision boundaries - get ${k \choose 2}$ boundaries where $\{x: f_k(x) = f_l(x)\}$.
*  We fit $min_B ||Y-XB||_F = min\sum_{k=1}^K ||y_k - X\beta_k||_2^2$ - have a coefficient for each of the responses instead of a coefficient vector. 
*  Note $E(Y|X=x)$ for a given one hot encoding Y for a class is the vector of conditional probabilities across possible classes.
*  We  could construct targets for each class $t_k$, and try to reproduce the correct target for an observation fit via OLS.
   * Could write the frobenius norm also as $\sum_{i=1}^N||y_i = B^Tx_i||^2_2$
   * Classifying to closest target equivalent to highest probability for class
*  As number of classes K increases, classes can be masked by others - linear model misses linear boundaries completely. Falls out as a curiosity of using least squares and a linear model for this problem.
   * Reduce this problem down to 1 dimension by projecting on a line through the three classes. The rug plot shows the fitted regression line relative to that line. The green and orange points have lines with non zero slopes - the likelihood varies. But the middle blue class has a line without slope and never is the most likely. The problem goes away if we fit quadratic functions, but we shouldn’t need quadratics because something like LDA can fit easily with linear functions. This is a problem as in higher dimensions we may not see we have this problem at all.
   * We shouldn’t have to introduce non-linearity to remove an artifact from our method - best to use a method without an artifact at all.
* A loose but general rule is that if K ≥ 3 classes are lined up, polynomial terms up to degree K − 1 might be needed to resolve them.

### Linear Discriminant Analysis

* From Bayes’ Theorem, $\operatorname{Pr}(G=k | X=x)=\frac{f_{k}(x) \pi_{k}}{\sum_{\ell=1}^{K} f_{\ell}(x) \pi_{\ell}}$
* $f_k(x)$ is our density when G = k. $\pi_k = Pr(G = k)$ - the marginal probability that G equals k. If we know both of these for 1,...,k, then we know everything about the joint distribution between G and X. $P_X(G=k) = P_X(X=x|G=k)P(G=k) = f_k(x)\pi_k$
* Then by Bayes, $P(G_k|X=x) = \frac{P_X(G=k, X=x)}{P(X=x)} = \frac{f_k(x)\pi_k}{\sum_{l=1}^kf_l(x)\pi_l}$ where the marginal is just summing over all k to get rid of it from the joint. However the density estimation is hard, and in high dimensions it is especially hard. Generative models - if you have the formula for the densities easy to generate the boundaries.
* Take the log ratio of the conditional probabilties - if this log ratio is 0, then we have equal probability and determines the decision boundary. Think of two normal bells overlapping - we take the boundary as the intersection of the densities - but notice that some will be misclassified where the density of one slips below the other.
* Suppose we model each class density as multivariate Gaussian. Linear discriminant analysis (LDA) arises in the special case when we assume that the classes have a common covariance matrix $\Sigma_k = \Sigma \; \forall k$. 
* MVN: $f_{k}(x)=\frac{1}{(2 \pi)^{p / 2}|\Sigma|^{1 / 2}} e^{-\frac{1}{2}\left(x-\mu_{k}\right)^{T} \Sigma^{-1}\left(x-\mu_{k}\right)}$ 
* In comparing two classes k and l, it is sufficient to look at the log-ratio, and we see that $\log \frac{\operatorname{Pr}(G=k | X=x)}{\operatorname{Pr}(G=\ell | X=x)} = \begin{array}{l}
  {\log \frac{\pi_{k}}{\pi_{\ell}}-\frac{1}{2}\left(\mu_{k}+\mu_{\ell}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mu_{k}-\mu_{\ell}\right)} 
  {\quad+x^{T} \boldsymbol{\Sigma}^{-1}\left(\mu_{k}-\mu_{\ell}\right)}
  \end{array} = \delta_{k}(x)-\delta_{\ell}(x)$ where $\delta_{k}(x)=\log \pi_{k}-\frac{1}{2} \mu_{k}^{T} \Sigma^{-1} \mu_{k}+x^{T} \Sigma^{-1} \mu_{k}$. The convenient assumptions give us a nice linear form for the function.
* Linear discriminant functions: $\delta_{k}(x)=x^{T} \boldsymbol{\Sigma}^{-1} \mu_{k}-\frac{1}{2} \mu_{k}^{T} \boldsymbol{\Sigma}^{-1} \mu_{k}+\log \pi_{k}$, equivalent to the decision rule using $G(x)=\operatorname{argmax}_{k} \delta_{k}(x)$
* This is the population model, but obviously we need to plug in estimates from our data. The covariance matrix needs to be invertible, then we multiply by $\mu_k$ and get linear coefficients for x. Have a sample $x_i,g_i, \; i \in 1,...,n$, for the covariance matrix need to take into account that it is the same estimate across all classes. Say $S_k = \frac{1}{n_k}\sum_{g_i=k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$ for $n_k$ obs in class k. Then $\hat{\Sigma} = \sum_{k=1}^K \pi_kS_k$ - pooled within class covariance matrix.
* Need to estimate the Gaussian parameters from the training data since they are unknown. 
  * $\hat{\pi}_{k}=N_{k} / N$
  * $\hat{\mu}_{k}=\sum_{g_{i}=k} x_{i} / N_{k}$
  * $\hat{\mathbf{\Sigma}}=\sum_{k=1}^{K} \sum_{g_{i}=k}\left(x_{i}-\hat{\mu}_{k}\right)\left(x_{i}-\hat{\mu}_{k}\right)^{T} /(N-K)$
* Choose the cut-point that empirically minimizes training error for a given dataset. For two classes, we can just subtract one discriminant function from the other and categorize based on positive or negative value, eg classify to class 2 if $x^{T} \hat{\Sigma}^{-1}\left(\hat{\mu}_{2}-\hat{\mu}_{1}\right)>\frac{1}{2}\left(\hat{\mu}_{2}+\hat{\mu}_{1}\right)^{T} \hat{\Sigma}^{-1}\left(\hat{\mu}_{2}-\hat{\mu}_{1}\right)-\log \frac{N_{2}}{N_{1}}$. Note that the discriminant direction minimizes overlap in distribution for Gaussian data - skew projection, see that the density overlap is minimized in this direction. The $\hat{\mu}_{2}-\hat{\mu}_{1}$ is a point between the two mean vectors then modulated by the relative prize $\log \frac{N_{2}}{N_{1}}$ - if one class is much bigger than the other, want to scale the density to favor the bigger class and minimize total number of misclassified data. The presence of $\Sigma$ makes it a skew projection instead of a standard projection.
* With more than two classes, LDA is not the same as linear regression of the class indicator matrix, and it avoids the masking problems associated with that approach
* We can enrich the LDA - augment the two variables with their squares and cross product - go from $\R^2 \rightarrow \R^5$. You get some weird correlation and you can project the linear $\R^5$ boundaries back down to $\R^2$ to get quadratic boundaries. They can perform pretty well. Of course, not really necessary since we can turn to QDA with more flexible assumptions.
* Without assumption of equal variance matrices, get QDA when the quadratic terms no longer cancel between discriminant functions. The estimates for QDA are similar to those for LDA, except that separate covariance matrices must be estimated for each class. When p is large this can mean a dramatic increase in parameters.
* Doing it with augmented LDA vs QDA - get very similar boundaries, or at very least similar misclassification rates. Generally look at the marginals of the variable, do they look roughly Gaussian, probably can fit this model. Otherwise perhaps look for other methods or transform the data to look more Gaussian (of course remember the transformation come test time) - Gaussiam copula model.

##### LDA computations

* Diagonalize $\hat{\mathbf{\Sigma}} \text { or } \hat{\mathbf{\Sigma}}_{k}$, using eigendecomposition $\hat{\mathbf{\Sigma}}_{k}=\mathbf{U}_{k} \mathbf{D}_{k} \mathbf{U}_{k}^{T}$

##### Regularized DA

* With big number of parameters, problematic to estimate covariance matrices
* QDA: shrink the separate covariances towards the common covariance matrix by parameter $\alpha$. For $\alpha=1$ get QDA and for $\alpha=0$ get LDA.
* May even be $\hat{\Sigma}$ is still too hard just for LDA. Shrink the pool covariance matrix towards a scalar covariance matrix - a scaled version of I, can only do if vars are in the same units. $\gamma$ parameter determines level of shrinkage, like a ridge version of DA.
* Vowel data - the full $\alpha=1$ has bigger misclassification rate, but just below 1 we get best performance. Regularization solving some instability in using separate covariance matrices.

##### Reduced Rank LDA

* The K centroids in p-dimensional input space lie in an affine subspace of dimension ≤ K − 1, and if p is much larger than K, this will be a considerable drop in dimension.
* Project $X^*$ onto the centroid spanning subspace since we only care about relative distances to centroids - we need only consider data in a subspace of dimension at most K-1
* We could make subspace even smaller by finding the PCs of the centroids themselves. Say we have three centroids that are nearly collinear - instead of projecting into $\R^3$, we aren’t going to lose much if we project onto $\R^1$ instead.
* Fisher: Find the linear combination $Z = a^TX$ st the between class variance is maximized relative to the within class variance. The between class variance is the variance of the class means of Z, and the within class variance is the pooled variance about the means.
* For W, common covariance matrix (within class covariance), the between-class variance of Z is $a^{T} \mathbf{B} a$ and within class variance is $a^{T} \mathbf{W} a$ where $B+W =T$, total covariance matrix of X. Fisher’s problem amounts to maximizing the Rayleigh quotient, $\max _{a} \frac{a^{T} \mathbf{B} a}{a^{T} \mathbf{W} a}$ 

##### Comparing DA Methods

* Say we have two classes, look at P(G=1|x) vs x. Can have a horrible bumpy function, but all we really want to know is where is it above 0.5. Estimating this with a smooth function will be highly biased - but could have the same classifications since below or above 0.5 is all that matters for this classification.

##### Example: Classification of Microarray Samples

* Four varieties of tumors - BL, EWS, NB, RMS - 4 classes. Small dataset but 2300 genes, very wide data. LDA not possible, pooled covariance is 2300 x 2300 - rank is very low.
* Row is gene, column sample, grouped by classes. We can see from the image there is some grouping within classes. The paper fit a NN, and way too many parameters. Used bagging on the NN which is why there are many learning curves. Ended with 0 training and test error. They fit the network on a projection onto PCs - Hastie got same performance with ridge logistic regression - similarly depends on PCs.
* Class centroid plot - centered the classes. Then with shrinkage, the blue lines are centroids that remained in the model for genes, while genes without blue shrunk out of model

##### Shrunken Centroids

* Have within class centroids and overall centroid
* For each class and component, take the deviation between component for kth class less the overall mean for variable j standardized by sj, the pooled variance for that gene.
* Shrinkage is soft-thresholding - just like Lasso. If positive, shrink it down by an amount delta, if negative shrink it up by delta. But if it crosses zero in the transform, it is set to 0.
* For a test sample $x^*$ with p components, the discriminant score for class k $\delta_{k}\left(x^{*}\right)=\log \pi_{k}-\frac{1}{2} \sum_{j=1}^{p} \frac{\left(x_{j}^{*}-\bar{x}_{j k}^{\prime}\right)^{2}}{s_{j}^{2}}$ then map to class with largest discriminant function. If for variable j, all shrunk to the overall centroid, then we can throw variable j out of the model since it contributes nothing new to the model.
* As shrinkage increases on genes, get better performance until a threshold where the genes really matter, then error shoots up with more shrinkage.
* The shrinkage also highlights the important genes, those that survived in the model. Interestingly, there were different sets of genes that survived for each of the classes.
* The ridge model worked very well, but a little inconvenient bc did not perform feature selection. This method focuses more on simplicity with feature selection.

### Logistic Regression

### Separating Hyperplanes