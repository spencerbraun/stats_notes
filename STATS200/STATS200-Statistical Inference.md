---
title: Statistical Inference Notes
date: 20191003
author: Spencer Braun
---

**Table of Contents**

[TOC]

### Chapter 1: Probability

* intersection - probability that both A and B occur
* Complement - $A^c$ event that A does not occur, all events in the sample space that are not A
* Disjoint - A and C are disjoint if $A \cap C = \empty$ 
* Probability Axioms: 1) $P(\Omega) = 1$, 2) If $A \subset \Omega$ then $P(A) \geq 0$ 3) If A, B disjoint then $P(A \cup B) - P(A) + P(B)$
* Addition Law: $P(A \cup B) = P(A) +P(B) - P(A \cap B)$
* Permutation: ordered arrangement of objects
* Binomial coefficients: $(a+b)^n = \sum_{k=0}^n a^kb^{n-k}$
* \# of ways n objects can be grouped into r classes with $n_I$ in the $i^{th}$ class: ${n\choose n_1n_2...n_r} = \frac{n!}{n_1!n_2!...n_r!}$
* Bayes, multiplcation law: $P(A|B) = \frac{P(A \cap B)}{P(B)}$,  $P(B_j|A) = \frac{P(A|B_j)P(B_j)}{\sum P(A|B_i)P(B_i)}$
* Law of total probability: $P(A) = \sum P(A|B_i)P(B_i)$
* Independence: $P(A \cap B) = P(A)P(B)$. Mutual independence implies pairwise independence

### Chapter 2: Random Variables

* CDF: $F(x) = P(X \leq x)$

##### Discrete Mass Functions

* Bernoulli: $p\left (x \right ) = \begin{Bmatrix}
  p^{x} \left ( 1-p \right )^{1-x}, & if \: x = 0 \: or \: x = 1
  \\ 
  0, & otherwise
  \end{Bmatrix}, 
  \: 0\leq p\leq 1$

* Binomial: $p(k)=\binom{n}{k}p^{k}(1-p)^{n-k}\\
  n \in \mathbb{N}, k=0,1,...,n, \,
  0\leq p\leq 1$

* Negative Binomial 
  $$
  P(X=k) = {k-1 \choose r-1}p^r (1-p)^{k-r}
  \quad \quad
  , 0 \leq p \leq 1,  k = r, r+1, \ldots,  r = 1, 2, \ldots, k
  $$

* Hypergeometric

  * *n*: population size; *n*∈N
  * *r*: successes in population; *r*∈{0,1,...,*n*}
  * *m*: number drawn from population; *m*∈{0,1,...,*n*}
  * *X*: number of successes in drawn group
  * $P(X=k)=\frac{\binom{r}{k}\binom{n-r}{m-k}}{\binom{n}{m}} \, \max(0,m+r-n) \leq k \leq \min(r,m) \, 0 \leq p(k) \leq 1$

* Poisson: $P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda}\, k = 0,1,2,3,...$

##### Continuous Density Functions

* Uniform: $f(x) = \begin{cases}   1/ (b - a) & a \leq x \leq b\\  0 & x < a \ or \ x > b \\  \end{cases} \,, x \in [a,b] \,,  a<b\, , f : \mathbb{R} \mapsto [0, \infty )$
* Exponential $f(x) = \left\{\begin{matrix}   \lambda e^{-\lambda x} , &  x \geq 0\\   0 , & x< 0 \end{matrix}\right. \, ,\, \lambda > 0$
* Normal: $f(x\mid \mu,\sigma^2)=\frac{1}{\sqrt{2 \pi}\sigma }e^{-\frac{(x-\mu)^2}{2\sigma^2}}, f:\mathbb{R}\to(0,\infty), \mu\in \mathbb{R},\sigma>0$
* Gamma: 
  * $\Gamma(x) = \int_0^\infty u^{x-1}e^{-u} \, du, x > 0$
  * $g(t \mid \alpha, \lambda) = \begin{cases} \frac{\lambda^\alpha}{\Gamma(\alpha)} t^{\alpha-1} e^{-\lambda t}, t \geq 0 \\ 0, t < 0 \end{cases}$
  * $g: \mathbb{R} \to [0,\infty), \alpha > 0, \lambda > 0$
* Beta: $f(u) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}u^{a-1}(1-u)^{b-1}, 0\leq u\leq 1, \,\, a, b > 0$ 
* Cauchy: $f_Z(z) = \frac{1}{\pi(z^2 + 1)}$ for $z \in (-\infty,\infty)$

##### Functions of RVs

* If $X∼N(\mu, \sigma^2) \text{and} Y = aX + b,\text{ then }Y∼N(a\mu + b, a^2\sigma^2 ).$
* **Change of variables for function of RV** $f_Y(y) = f_X(g^{-1}(y))\Big|\frac{d}{dy}g^{-1}(y)\Big|$
* Universality of uniform: Z = F(X) for $Z \sim U(0,1)$. $X = F^{-1}(U)$ then $F(x) = P(U \leq F(x))$

### Chapter 3: Joint Distributions

* Marginal frequency: $p_X(x) = \sum_i p(x, y_i)$
* Joint density, $f(x, y)$, then $F(x, y) = \int_{-\infty}^x \int_{-\infty}^y f(u,v) dv\,du$
* Marginal cdf $F_X(x) = \int_{-\infty}^x \int_{-\infty}^\infty f(u,y) dy\,du$, marginal density $f_X(x) = \int_{-\infty}^\infty f(x, y) dy$
* Copula: a joint CDF of RVs that have uniform marginal distributions
* Random variables $X_1,X_2,...,X_n$ are independent if the joint CDF factors into the product of marginals: $F(x_1,x_2,...,x_n) = F_{X_1}(x_1)F_{X_2}(x_2)...F_{X_n}(x_n)$
* Conditional density: $f_{Y|X}(y|x) = \frac{f_{XY}(x,y)}{f_X(x)}$
* Convolution: X, Y RVs, Z = X + Y, then density of Z found by noting $Z=z$ when $X=x$ and $Y=z-x$. Therefore $f_Z(z) = \int_{-\infty}^\infty f(x, z-x) dx$ is the convolution of $f_X, f_Y$

###### Variable Transformation

* $u = g_1(x,y), \, v=g_2(x,y)$ for X, Y jointly distributed RVs
* Invert the transformation to $x = h_1(u,v), \, y=h_2(u,v)$
* Take jacobian wrt u and v for x and y ie. $\frac{\partial x}{\partial u}$
* Then change of variables = $f_{UV}(u, v) = f_{XY}(h_1(u,v), h_2(u,v))|J|$

### Chapter 4: Expected Values

* $E(X)=\int_{-\infty}^{\infty} x f(x) d x$ provided the expectation of |x| exists
* Functions of RVs: $E(g(X)=\int_{-\infty}^{\infty} g(x) f(x) d x$ - do not need the pdf of g(x) for expectation
* For independent X and Y, E(XY) = E(X)E(Y).
* Linearity of expectation - constants and coefficients can be pulled out of expectation
* Variance
  * $\operatorname{Var}(X)=\sum_{i}\left(x_{i}-\mu\right)^{2} p\left(x_{i}\right)$
  * $\operatorname{Var}(X)=E\left\{[X-E(X)]^{2}\right\}$
  * $\operatorname{Var}(X)=E\left(X^{2}\right)-[E(X)]^{2}$
  * $\operatorname{Var}(X)=Cov(X, X)$
  * $\operatorname{Cov}(X, Y)=E\left[\left(X-\mu_{X}\right)\left(Y-\mu_{Y}\right)\right] = E(XY) - E(X)E(Y)$. If X,Y independent, Cov = 0
  * $\operatorname{Cov}(aX, bY)=abCov(X, Y)$
  * $\begin{aligned} \operatorname{Var}(X+Y) &=\operatorname{Cov}(X+Y, X+Y) \\ &=\operatorname{Var}(X)+\operatorname{Var}(Y)+2 \operatorname{Cov}(X, Y) \end{aligned}$
* Conditional Expectation: $E(Y | X=x)=\int y f_{Y | X}(y | x) d y$
* Law of total expectation: E(Y) = E(E(Y|X)), $\operatorname{Var}(Y)=\operatorname{Var}[E(Y | X)]+E[\operatorname{Var}(Y | X)]$

###### Markov Inequality

* X RV with domain geq 0, then $P(X \geq t) \leq \frac{E(X)}{t}$
* This result says that the probability that X is much bigger than E(X) is small. 

###### Chebyshev’s Inequality

* $P(|X-\mu|>t) \leq \frac{\sigma^{2}}{t^{2}}$
* Plug in t = K x sigma to get alternate form

###### Moment Generating Functions

* Uniquely determines a probability distribution
* $M(t) = E(e^{tX})= \int_{-\infty}^\infty e^{tx} f(x) \, dx$ 
* The rth moment: $M^{(r)}(0) = E(X^r)$
* $\text { If } X \text { has the } \operatorname{mgf} M_{X}(t) \text { and } Y=a+b X, \text { then } Y \text { has the mgf } M_{Y}(t)=e^{a t} M_{X}(b t)$
* If X, Y independent and Z= X + Y, Mz(t) = Mx(t)My(t)

###### delta Method

* For Y = g(x)
* $\sigma_{Y}^{2} \approx \sigma_{X}^{2}\left[g^{\prime}\left(\mu_{X}\right)\right]^{2}$ and $E(Y) \approx g\left(\mu_{X}\right)+\frac{1}{2} \sigma_{X}^{2} g^{\prime \prime}\left(\mu_{X}\right)$

### Chapter 5: Limit Theorems

##### Weak Law of Large Numbers (Convergence in probability)

* $E(X_i) = \mu $ and $ Var(X_i) =\sigma^2$ . Let $X_n = n^{−1} \sum_{i=1}^n X_i$
* $P(|\bar{X_n} - \mu| > \epsilon) \rightarrow 0$ as $n \rightarrow \infty$
* $X_1 ... X_i \sim iid$
* Need to have  constant variance to apply WLLN -> otherwise must normalize across $X_i$s

##### Convergence in Distribution

* $X_1, X_2$ sequence of RVs with CDFs $F_!, F_2...$. Let X be RV with distribution function F. Say $X_n$ converges in distribution to X if:
* $\lim_{n\to \infty} F_n(x) = F(x)$ at every point for continuous F
* MGFs usually used to show this property. 

##### Continuity Theorem

* $F_n$ be a sequence of CDFs with MGF $M_n$. F CDF with MGF M. 
* If $M_n(t) \rightarrow M(t)$ for all t then $F_n(x) \rightarrow F(X)$ for all points of F

##### Central Limit Thm

* $X_1, X_2$ sequence of RVs with mean 0 and variance $\sigma^2$ common CDF F and MGF M.
* $S_n = \Sigma_{i=1}^n X_i$ 
* $\lim_{n\to \infty} P(\frac{S_n}{\sigma \sqrt(n)} \leq x) = \Phi(x)$ on $-\infty < x < \infty$

### Chapter 6: Distributions Derived from the Normal

##### Chi Square Distribution

* U = $Z^2$ for $Z \sim N(0,1)$ , then $U \sim \chi^2_1$ 
* Note that the distribution is equivalent to $\chi^2_1 \sim \Gamma(\frac{n}{2}, \frac{1}{2})$
* If U1, U2, ... ,Un are independent chi-square random variables with 1 degree of  freedom, the distribution of V = U1 + U2 + ··· + Un is called the chi-square distribution with n degrees of freedom and is denoted by $\chi^2_n$

##### t-distribution

* $Z \sim N(0,1)$ and  $U \sim \chi^2_n$ for Z, U independent, then 
* $\frac{Z}{\sqrt{\frac{U}{n}}}$ is a t-distribution with n degrees of freedom
* density function of $$f(t) = \frac{\Gamma[\frac{1}{2}(n+1)]}{\sqrt{n\pi}\Gamma(n/2)}\Big(1 + \frac{t^2}{n}\Big)^{-\frac{n+1}{2}}$$

##### F Distribution

* U, V independent chi-square RVs wit m and n respective DoF
* $W = \frac{U/M}{V/n}$ is F with m and n DoF, ie. $F_{m,n}$

##### Sample Statistics

* For $X_1, ..., X_n$ iid normals, we sometimes refer to them as a sample from a normal distribution.
* $\bar{X} = \frac{1}{n} \sum_{i=1}^n X_i$ = sample mean
* $S^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2$ = sample variance
* Note that $E(\bar{X}) = \mu$ and $Var(\bar{X}) = \frac{\sigma^2}{n}$
* The RV X-bar and the vector of RVs $(X_1 - \bar{X}, ...)$ are independent
* $\bar{X}$ and $S^2$ are independently distributed
* The distribution of $\frac{(n−1)S^2}{\sigma^2} \sim \chi^2_{n-1} $
* Important: $\frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}$

### Chapter 7: Survey Sampling

### Chapter 8: Estimation of Parameters and Fitting Distributions

* The observed data will be regarded as realizations of random variables $X_1,X_2,...,X_n$,  whose joint distribution depends on an unknown parameter $\theta$. An estimate of $\theta$ will be a function of  $X_1,X_2,...,X_n$,   and will hence be a random variable with a probability distribution called itssampling  distribution.
* Standard Error: the standard deviation of the estimate of a parameter. $\sigma_{\hat{\theta}} = \sqrt{\frac{\theta_0}{n}}$. We generally do not know the true standard error, since it is a function of the true parameter.
* Estimated standard error - use the estimate of parameter: $s_{\hat{\theta}} = \sqrt{\frac{\hat{\theta}}{n}}$
* If $E(\hat{\lambda}) = \lambda$ then we say the estimate is unbiased. 
* An estimate $\hat{\theta}$ is said to be consistent if $\hat{\theta} \rightarrow \theta$ as $n \rightarrow \infty$. More precisely: $P(|\hat{\theta_n} - \theta| > \epsilon) \rightarrow 0$ as $n \rightarrow \infty$. Note convergence in probability 
* 

##### Method of Moments

* kth sample moment defined as $\hat{\mu_k} = \frac{1}{n}\sum_{i=1}^nX_i^k$, where mu-hat is an estimate of the kth moment.
* If two parameters $\theta_1, \theta_2$ can be expressed in terms of the first two moments as $\theta_1 = f_1(\mu_1, \mu_2), \theta_2 = f_2(\mu_1, \mu_2)$ then the method of moments estimates are $\hat{\theta_1} = f_1(\hat{\mu_1}, \hat{\mu_2}), \hat{\theta_2} = f_2(\hat{\mu_1}, \hat{\mu_2})$ 
* Steps
  * Calculate low order moments, finding expressions for the moments in terms of the parameters. Usually need the same number of moments as parameters
  * Invert the expressions, finding parameters in terms of moments
  * Insert sample moments into the above expressions and you have your parameter estimates.

##### Maximum Likelihood Estimation

* Random variables $X_1,X_2,...,X_n$ have a joint density or frequency function $f (x_1, x_2,..., x_n|\theta)$. Given observed values $X_i = x_i$ , where $i = 1,..., n$  the likelihood of $\theta$ as a function of $x_1, x_2,..., x_n$ is defined as  $lik(\theta) = f (x_1, x_2,..., x_n|\theta)$.
* The MLE of $\theta$ is that value that maximizes the likelihood of f - it makes the observed data most probable
* For iid X, $lik(\theta) = \prod_{i=1}^nf(X_i|\theta)$. Can also use the log likelihood: $l(\theta) = \prod_{i=1}^nlog[f(X_i|\theta)]$
* Steps
  * Find joint density function, viewed as a function of the parameters.
  * Take partials with respect to the parameters (eg. $\mu, \sigma$). Set partials to zero and solve for parameters.
  * Those parameters are the estimates - may need to solve system of equations to get RHS of equations just in terms of X's

###### Large Sample Theory for MLE

* The MLE from an iid sample is consistent (given smoothness of f)
* Define $I(\theta) = E\Big[\frac{\partial}{\partial\theta}logf(X|\theta) \Big]^2$. This can also be expressed as $I(\theta) = -E\Big[\frac{\partial^2}{\partial\theta^2}logf(X|\theta) \Big]^2$
* Under smoothness conditions on f ,the probability distribution of $\sqrt{nI(\theta_0)}(\hat{\theta} −\theta_0)$  tends to a standard normal distribution. We say the MLE is asymptotlically unbiased.
* Can also be interpreted as for an MLE from the log-likelihood function $l(\theta)$, the asymptotic variance is $\frac{1}{nI(\theta_0)} = -\frac{1}{El’’(\theta_0)}$

###### CI from MLE

* 3 methods: exact. approximations with large samples, and bootstrap CI
* Exact example:
  * For $S^2 = \frac{1}{n-1}\sum_{i=1}^n(X_i - \bar{X})^2$, we have $\frac{\sqrt{n}(\bar{X} - \mu)}{S} \sim t_{n-1}$
  * Then CI for $\mu$ = $P( \bar{X}\ - \frac{S}{\sqrt{n}}t_{n-1}(\alpha/2)\leq \mu \leq \bar{X} + \frac{S}{\sqrt{n}}t_{n-1}(\alpha/2)) = 1- \alpha$ 
  * Given $\frac{n\hat{\sigma}^2}{\sigma^2} \sim \chi^2_{n-1}$ CI for $\sigma = P(\frac{n\hat{\sigma}^2}{\chi^2_{n-1}(\alpha/2)} \leq \sigma^2 \leq \frac{n\hat{\sigma}^2}{\chi^2_{n-1}(1-\alpha/2)}) = 1- \alpha$. Note this is not symmetric about $\hat{\sigma}^2$
* Approximate
  * $\sqrt{nI(\hat{\theta})}(\hat{\theta} −\theta_0)$  tends to a standard normal distribution
  * There $P((\alpha/2) \leq \sqrt{nI(\hat{\theta})}(\hat{\theta} −\theta_0) \leq z(\alpha/2)) \approx 1 - \alpha$
  * This gives us $\hat{\theta} +- z(\alpha/2)\frac{1}{\sqrt{nI(\hat{\theta})}} $
  * Similarly, $Var(\hat{\theta}) \approx \frac{1}{El’(\theta_0)^2} = -\frac{1}{El’’(\theta_0)}$ and the MLE is approximately normally distributed
* Bootstrap
  * Assume we know the distribution of $\delta = \hat{\theta} - \theta_0$. Denote $\alpha/2$ and $1-\alpha/2$ as quantiles of this distribution by $\underline{\delta}, \, \bar{\delta}$
  * $P(\underline{\delta}\leq \hat{\theta} - \theta_0 \leq \bar{\delta}) = 1 - \alpha$  so $P(\hat{\theta} - \bar{\delta}\leq  \theta_0 \leq  \hat{\theta} - \underline{\delta}) = 1 - \alpha$ 
  * If $\theta_0$ were known, this distribution could be approximated  arbitrarily well by simulation. Since $\theta_0$ is not known, the bootstrap principle suggests using $\hat{\theta}$ in its place: Generate many, many samples (say, B in all) from a distribution with value $\hat{\theta}$; and for each sample construct an estimate of $\theta$.

##### Bayesian Estimation

* Unknown parameter is treated as an RV with a prior distribution $f_\Theta(\theta)$ what we know about the parameter before observing data X. Assume $\Theta$ is a continuous RV.
* Therefore we can get the joint distribution of X and $\Theta$ by $f_{X,\Theta}(x, \theta) = f_{X|\Theta}(x|\theta)f_\Theta(\theta)$
* And we can use Bayes rule to get the posterior proportional to likelihood times prior : $f_{\Theta|X}(\theta|x) \propto f_{X|\Theta}(x|\theta)f_\Theta(\theta)$
* Essentially $f_{\Theta|X}(\theta|x) = \frac{f_{X,\Theta}}{f_x(x)} = \frac{f_{X|\Theta}(x|\theta)f_\Theta(\theta)}{\int f_{X|\Theta}(x|\theta)f_\Theta(\theta) \, d\theta}$
* Steps
  * Find the prior distribution
  * Integrate of the denominator of Bayes, ie $\int f_{X|\Theta}(x|\theta)f_\Theta(\theta) \, d\theta$
  * Divide to get the posterior
  * Note: Now, consider this an important trick that is used time and again in Bayesian calculations: the denominator is a constant that makes the expression integrate to 1. We can deduce from the form of the numerator that the ratio must be a gamma density
* High posterior density interval

##### Cramer-Rao Lower Bounds / Efficiency

* Given two estimates $\hat{\theta}, \bar{\theta}$ of the same parameter, the efficiency of $\hat{\theta}$ relative to $\bar{\theta}$ is defined to be $eff(\hat{\theta, \bar{\theta}}) = \frac{Var(\bar{\theta})}{Var(\hat{\theta})}$
* If eff is smaller than 1, then theta-hat has a larger variance than theta-bar. Most meaningful when the estimates have the same bias. if Var(estimator) = $\frac{c}{n}$, then the efficiency is the ratio os sample sizes necessary to obtain the same variance for both estimators.
* Cramer Rao inequality: $X_1,X_2,...,X_n$ iid with density function $f(x|\theta)$. Let $T = t(X_1,X_2,...,X_n)$ be an unbiased estimate of $\theta$. Then $Var(T) \geq \frac{1}{nI(\theta)}$
* Theorem A gives a lower bound on the variance of any unbiased estimate. An  unbiased estimate whose variance achieves this lower bound is said to be efficient.  Since the asymptotic variance of a maximum likelihood estimate is equal to the  lower bound, maximum likelihood estimates are said to be asymptotically efficient. 

##### Sufficiency

* Given the initial conditions for Cramer Rao above. 
* The concept of sufficiency arises as an attempt  to answer the following question: Is there a statistic, a function $T (X_1,X_2,...,X_n)$, that  contains all the information in the sample about $\theta$? If so, a reduction of the original  data to this statistic without loss of information is possible. For example, in series of Bernoulli trials, total numbr of successes contains all of the information about p that exists in the sample.
* Sufficiency: A statistic $T (X_1,X_2,...,X_n)$ is said to be sufficient for $\theta$ if the conditional distribution of $X_1,X_2,...,X_n$ given $T = t$, does not depend on $\theta$ for any value  of t.
* In other words, given the value of T , which is called a sufficient statistic, we can  gain no more knowledge aboutθfrom knowing more about the probability distribution  of $X_1,X_2,...,X_n$
* Factorization: A necessary and sufficient condition for $T (X_1,X_2,...,X_n)$ to be sufficient for a  parameter  $\theta$ is that the joint probability function (density function or frequency  function) factors in the form: $f(x_1,...,x_n|\theta) = g[T(x_1,...,x_n), \theta]h(x_1,...,x_n)$

###### Rao Blackwell Theorem

* Let $\hat{\theta}$ be an estimator of $\theta$ with existing expectation for all theta. If T is sufficient for theta and $\bar{\theta} = E(\hat{\theta}|T)$. Then for all theta: $E(\bar{\theta} - \theta) \leq E(\hat{\theta} - \theta)^2$
* Rationale for basing estimators on sufficient stats if they exist. 