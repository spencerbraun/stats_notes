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

### Statistical Models

* Additive error model for regression $Y=f(X)+\varepsilon$ where random error epsilon has $E(\varepsilon) = 0$, independent of X. $f(x)=\mathrm{E}(Y | X=x)$
* Generally there will be other unmeasured variables that also contribute to Y, including measurement error. The additive model assumes that we can capture all these departures from a deterministic relationship via the error epsilon.
* Function approximation - The goal is to obtain a useful approximation to f(x) for all x in some region of $\R^p$, given the representations in T. Allows us to use geometrical concepts and probabilistic inference.
* Linear basis expansions $f_{\theta}(x)=\sum_{k=1}^{K} h_{k}(x) \theta_{k}$. 
  * $h_k$ are a suitable set of functions or transformations of the input vector x, such as polynomials, trig functions. Or non-linear expansions such as the sigmoid $h_{k}(x)=\frac{1}{1+\exp \left(-x^{T} \beta_{k}\right)}$
  * $\theta$ - set of parameters modified to suit the data at hand, say $\theta = \beta$ for regression. Could estimate through OLS
* We imagine our parameterized function as a surface in p+1 space and we just see noisy realizations of it.
* MLE - for random sample, density $y_{i}, i=1, \ldots, N,\; \operatorname{Pr}_{\theta}(y)$, get log probability $L(\theta)=\sum_{i=1}^{N} \log \operatorname{Pr}_{\theta}\left(y_{i}\right)$
  * For additive model, LS equivalent to MLE using conditional likelihood $\operatorname{Pr}(Y | X, \theta)=N\left(f_{\theta}(X), \sigma^{2}\right)$

### Structured Regression Models

* For arbitrary f, minimizing $\operatorname{RSS}(f)=\sum_{i=1}^{N}\left(y_{i}-f\left(x_{i}\right)\right)^{2}$ has infinite solutions since any f passing through the points is min.
* However If there are multiple observation pairs $x_i, y_{i\ell},  \;\ell= 1,...,N_i$ at each value of $x_i$, the risk is limited - solutions pass through average of $y_{i\ell}$. If sample size is infinite, this is guaranteed, but for finite N, need to reduce set of f’s.
* There are still infinite possible restrictions, so the amiguity still exists. In general the constraints imposed by most learning methods can be described as complexity restrictions of one kind or another. This usually means some kind of regular behavior in small neighborhoods of the input space. How small a neighborhood do we draw within which the function is constant.
* Any method that attempts to produce locally varying functions in small isotropic neighborhoods will run into problems in high dimensions—again the curse of dimensionality. And conversely, all methods that overcome the dimensionality problems have an associated—and often implicit or adaptive—metric for measuring neighborhoods, which basically does not allow the neighborhood to be simultaneously small in all directions.

### Classes of Restricted Estimators

* Roughness penalty and Bayesian methods
  * Penalize RSS with a roughness penalty: $\operatorname{PRSS}(f ; \lambda)=\operatorname{RSS}(f)+\lambda J(f)$
  * $J(f)$ will be large for functions that vary too rapidly over small regions of input space. Eg. cubic smoothing spline $\operatorname{PRSS}(f ; \lambda)=\sum_{i=1}^{N}\left(y_{i}-f\left(x_{i}\right)\right)^{2}+\lambda \int\left[f^{\prime \prime}(x)\right]^{2} d x$
  * Penalty function, or regularization methods, express our prior belief that the type of functions we seek exhibit a certain type of smooth behavior, and indeed can usually be cast in a Bayesian framework.
* Kernel Methods and Local Regression
  * Explicitly providing estimates of the regression function or conditional expectation by specifying the nature of the local neighborhood, and of the class of regular functions fitted locally.
  * Kernel function specifies local neighborhood $K_{\lambda}\left(x_{0}, x\right)$. Assigns weights to points x in region around $x_0$. 
  * Eg. local regression minimizing $\operatorname{RSS}\left(f_{\theta}, x_{0}\right)=\sum_{i=1}^{N} K_{\lambda}\left(x_{0}, x_{i}\right)\left(y_{i}-f_{\theta}\left(x_{i}\right)\right)^{2}$
* Basis Functions and Dictionary Methods
  * Includes the familiar linear and polynomial expansions, but more importantly a wide variety of more flexible models. Model f is a linear expansion of basis functions $f_{\theta}(x)=\sum_{m=1}^{M} \theta_{m} h_{m}(x)$. 
  * Radial basis functions are symmetric p-dimensional kernels located at particular centroids: $f_{\theta}(x)=\sum_{m=1}^{M} K_{\lambda_{m}}\left(\mu_{m}, x\right) \theta_{m}$

### Model Selection and Bias-Variance Tradoff

* $\operatorname{EPE}_{k}\left(x_{0}\right)=\mathrm{E}\left[\left(Y-\hat{f}_{k}\left(x_{0}\right)\right)^{2} | X=x_{0}\right] = \sigma^{2}+\left[\operatorname{Bias}^{2}\left(\hat{f}_{k}\left(x_{0}\right)\right)+\operatorname{Var}_{\mathcal{T}}\left(\hat{f}_{k}\left(x_{0}\right)\right)\right]=\sigma^{2}+\left[f\left(x_{0}\right)-\frac{1}{k} \sum_{\ell=1}^{k} f\left(x_{(\ell)}\right)\right]^{2}+\frac{\sigma^{2}}{k}$ for $ \mathcal{T}$= training data. $Y|x_0$ is independent of $\mathcal{T}$. $\sigma^2_{x_0}$ is the variance of Y arouund x0.
* Note $\hat{f}(X_0)$ is a RV, it depends on the training data - this leads directly to our understanding of bias and variance 
* 
* The first term $\sigma^2$ is the irreducible error. The second and third terms are under our control, and make up the mean squared error, broken down into a bias component and a variance component. 
* Bias: $\left[\mathrm{E}_{\mathcal{T}}\left(\dot{f}_{k}\left(x_{0}\right)\right)-f\left(x_{0}\right)\right]^{2}$ square difference between true mean and expected value of the estimate.
* The variance term is simply the variance of an average here, and decreases as the inverse of k.
* Often prediction error (EPE) is a vehicle for getting at MSE, which tells how well we are estimating f(x), our real interest.
* Performance of OLS vs KNN in Bias Variance Terms
  * OLS: $EPE(X_0) \approx \sigma^2 \times \frac{P}{N}\sigma^2 + bias^2_{OLS}(x_0)$
    * How did we get P/N: $\hat{\beta} = (X^TX)^{-1}X^Ty$ and $\hat{f} = X\hat{\beta}$. Say $Y \sim f(x) + \epsilon$ for $\epsilon \sim (0,\sigma^2)$. Treat X’s in training sample as fixed, condition on those X’s, then $\hat{f} = X\hat{\beta}=Ay$. $Cov(Y) = \sigma^2I_n$ so $Cov(\hat{f}) = \sigma = A\sigma^2$. What is the avg variance of $\hat{f}_i$? $Var(\hat{f}_i) = tr(A)\sigma^2/n$ where $tr(A) = p =$ # of parameters. So this is close to P/N sigma squared
  * KNN: $EPE(X_0) \ge \sigma^2 + \sigma^2 + bias^2_{1-NN}(x_0)$
    * Sigma squared - both irreducible error and the error from the closest neighbor. 
  * Concluding $\sigma^2 + P\sigma^2/N \approx \sigma^2 < 2\sigma^2$ - OLS better on variance. f(x) determines the bias, but for large N KNN could be optimal.

## Chapter 3: Linear Methods for Regression

