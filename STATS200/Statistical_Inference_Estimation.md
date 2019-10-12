---
title: Statistical Inference - Estimation
author: Spencer Braun
date: 20191009
---

[TOC]



### Chapter 7: Survey Sampling

##### Population Parameters

* N = population size
* All $x_i$ are numerical values from the population
* Then population mean = $\frac{1}{N}\sum_{i=1}^N x_i$
* t = $\sum_{i=1}^N x_i = N \mu$
* Population variance: $\sigma^{2}=\frac{1}{N} \sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}$

##### Sampling Parameters

* sample size n, values of the sample are $X_i$
* sample mean: $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$. This is a random variable
* Confidence interval for a normal sample: $P\left(-z(\alpha / 2) \leq \frac{\bar{X}-\mu}{\sigma_{\bar{X}}} \leq z(\alpha / 2)\right) \approx 1-\alpha$
* $P\left(\bar{X}-z(\alpha / 2) \sigma_{\bar{X}} \leq \mu \leq \bar{X}+z(\alpha / 2) \sigma_{\bar{X}}\right) \approx 1-\alpha$

### Chapter 8: Estimation of Parameters and Fitting Distributions

- The observed data will be regarded as realizations of random variables $X_1,X_2,...,X_n$,  whose joint distribution depends on an unknown parameter $\theta$. An estimate of $\theta$ will be a function of  $X_1,X_2,...,X_n$,   and will hence be a random variable with a probability distribution called itssampling  distribution.
- There are three different kinds of $\theta$ in this setting. First there is $\theta$ the parameter which has a range of legal values. Then there is $\theta_0$. When we need to single out the one true value of $\theta$ it is $\theta_0$. In practice we don’t know which value is true. Our MLE is $\hat{\theta}$.
- Bayesian estimation always treats a parameter as a random variable. Frequentist estimation sees the parameters as an unobserved value for which there is a true value.
- The set $\Omega$ of all possible values of a parameter $\theta$ or of a vector of parameters $(\theta_1,..., \theta_k)$ is called the parameter space.

- Standard Error: the standard deviation of the estimate of a parameter. $\sigma_{\hat{\theta}} = \sqrt{\frac{\theta_0}{n}}$. We generally do not know the true standard error, since it is a function of the true parameter.
- Estimated standard error - use the estimate of parameter: $s_{\hat{\theta}} = \sqrt{\frac{\hat{\theta}}{n}}$
- If $E(\hat{\lambda}) = \lambda$ then we say the estimate is unbiased. 
- An estimate $\hat{\theta}$ is said to be consistent if $\hat{\theta} \rightarrow \theta$ as $n \rightarrow \infty$. More precisely: $P(|\hat{\theta_n} - \theta| > \epsilon) \rightarrow 0$ as $n \rightarrow \infty$. Note convergence in probability 

##### Fisher Information

* This measure has the intuitive properties that more data provide more information, and more precise data provide more information. The variance of the distribution tends to be inversely proportional to I. 
* Consider a random variable X for which the p.f. or the p.d.f. is $f(x|\theta)$. It is assumed that $f(x|\theta)$ involves a parameter $\theta$ whose value is unknown but must lie in a given open interval $\Omega$ of the real line. Furthermore, it is assumed that X takes values in a specified sample space S, and  $f(x|\theta)$ > 0 for each value of x $\in$ S and each value of $\theta \in \Omega$ .
* Define $\lambda(x | \theta)=\log f(x | \theta)$ - log likelihood function
* The Fisher information = $I(\theta)=E_{\theta}\left\{\left[\lambda^{\prime}(X | \theta)\right]^{2}\right\} = -E_{\theta}\left[\lambda^{\prime \prime}(X | \theta)\right] =\operatorname{Var}_{\theta}\left[\lambda^{\prime}(X | \theta)\right]$. Can make it a function with respect to n and it is a sample statistic. 
* Staring into it we see it is an expected squared slope of log likelihood. If the slope is large then small changes in $\theta$ change the log likelihood a lot. That should help separate likely from unlikely values. If that slope were zero then we get no effect of changing $\theta$.
* $I_n(\theta) = nI(\theta)$ the Fisher information in a random sample of n observations is simply n times the Fisher information in a single observation.
* Can be used to compare sampling plans. Calculating the Fisher information for each and equating them will tell you something about the necessary parameters to yield the same information.
* Fisher information can be used to determine a lower bound for the variance of an arbitrary estimator of the parameter θ in a given problem -> Cramer - Rao


##### Method of Moments

- kth sample moment defined as $\hat{\mu_k} = \frac{1}{n}\sum_{i=1}^nX_i^k$, where mu-hat is an estimate of the kth moment.
- If two parameters $\theta_1, \theta_2$ can be expressed in terms of the first two moments as $\theta_1 = f_1(\mu_1, \mu_2), \theta_2 = f_2(\mu_1, \mu_2)$ then the method of moments estimates are $\hat{\theta_1} = f_1(\hat{\mu_1}, \hat{\mu_2}), \hat{\theta_2} = f_2(\hat{\mu_1}, \hat{\mu_2})$ 
- Steps
  - Calculate low order moments, finding expressions for the moments in terms of the parameters. Usually need the same number of moments as parameters
  - Invert the expressions, finding parameters in terms of moments
  - Insert sample moments into the above expressions and you have your parameter estimates.
- MOM estimator for variance: $\hat{\sigma}^{2}=\frac{1}{n} \sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)^{2}$ 

##### Conjugates

* For each of the most popular statistical models, there exists a family of distributions for the parameter with a very special property. If the prior distribution is chosen to be a member of that family, then the posterior distribution will also be a member of that family. Such a family of distributions is called a conjugate family.
* No matter the prior chosen, number of observations, if the posterior ends up as a member of a subset of distributions, we say it is a conjugate family of prior distributions for samples from $f(x|\theta)$
* Those conjugates may be parametetrized, and those parameters are the hyperparameters of the distribution (either prior or posterior)
* Beta is the conjugate family for Bernoulli -> if prior is Beta then the posterior will be too
* Gamma is a conjugate family of priors for Poisson
* Normals are conjugates of normals with unknown mean and known variance
* Gamma is conjugate of Posterior for exponential with known parameter

##### Maximum Likelihood Estimation

- Random variables $X_1,X_2,...,X_n$ have a joint density or frequency function $f (x_1, x_2,..., x_n|\theta)$. Given observed values $X_i = x_i$ , where $i = 1,..., n$  the likelihood of $\theta$ as a function of $x_1, x_2,..., x_n$ is defined as  $lik(\theta) = f (x_1, x_2,..., x_n|\theta)$. When the joint p.d.f. or the joint p.f. $fn(x|\theta)$ of the observations in a random sample is regarded as a function of $\theta $ forgiven values of $x_1,...,x_n$, it is called the likelihood function.
- The MLE of $\theta$ is that value that maximizes the likelihood of f - it makes the observed data most probable. For large samples, MLE often yields a very good estimator. It is a value we believe the parameter to be near, but other estimates are likely to be better with smaller samples or any prior information. MLE is range respecting unlike MOM - we won’t get an estimate that is beyond the domain of the parameter.
- If $\hat{\theta} $ is the maximum likelihood estimator of $\theta$ and if g is a one-to-one function, then $g(\hat{\theta})$ is the maximum likelihood estimator of $g(\theta)$. Or, if not one to one,, if we define $g(\theta)$ to be a function of $\theta$, then $g(\hat{\theta})$ is an MLE of  $g(\theta)$

- Therefore, the maximum likelihood estimate is the value of θ that assigned the highest probability to seeing the observed data. It is not necessarily the value of the parameter that appears to be most likely given the data.

- For iid X, $lik(\theta) = \prod_{i=1}^nf(X_i|\theta)$. Can also use the log likelihood: $l(\theta) = \prod_{i=1}^nlog[f(X_i|\theta)]$
- Steps
  - Find joint density function, viewed as a function of the parameters.
  - Take partials with respect to the parameters (eg. $\mu, \sigma$). Set partials to zero and solve for parameters.
  - Those parameters are the estimates - may need to solve system of equations to get RHS of equations just in terms of X's

###### Large Sample Theory for MLE

- The MLE from an iid sample is consistent (given smoothness of f)
- Define $I(\theta) = E\Big[\frac{\partial}{\partial\theta}logf(X|\theta) \Big]^2$. This can also be expressed as $I(\theta) = -E\Big[\frac{\partial^2}{\partial\theta^2}logf(X|\theta) \Big]$
- Under smoothness conditions on f ,the probability distribution of $\sqrt{nI(\theta_0)}(\hat{\theta} −\theta_0)$  tends to a standard normal distribution. We say the MLE is asymptotlically unbiased.
- Can also be interpreted as for an MLE from the log-likelihood function $l(\theta)$, the asymptotic variance is $\frac{1}{nI(\theta_0)} = -\frac{1}{El’’(\theta_0)}$

###### CI from MLE

- 3 methods: exact. approximations with large samples, and bootstrap CI
- Exact example:
  - For $S^2 = \frac{1}{n-1}\sum_{i=1}^n(X_i - \bar{X})^2$, we have $\frac{\sqrt{n}(\bar{X} - \mu)}{S} \sim t_{n-1}$
  - Then CI for $\mu$ = $P( \bar{X}\ - \frac{S}{\sqrt{n}}t_{n-1}(\alpha/2)\leq \mu \leq \bar{X} + \frac{S}{\sqrt{n}}t_{n-1}(\alpha/2)) = 1- \alpha$ 
  - Given $\frac{n\hat{\sigma}^2}{\sigma^2} \sim \chi^2_{n-1}$ CI for $\sigma = P(\frac{n\hat{\sigma}^2}{\chi^2_{n-1}(\alpha/2)} \leq \sigma^2 \leq \frac{n\hat{\sigma}^2}{\chi^2_{n-1}(1-\alpha/2)}) = 1- \alpha$. Note this is not symmetric about $\hat{\sigma}^2$
- Approximate
  - $\sqrt{nI(\hat{\theta})}(\hat{\theta} −\theta_0)$  tends to a standard normal distribution. Notice $\sqrt{nI(\hat{\theta})}$ is the theoretical limit SD from Cramer-Rao for efficient estimators - however it can work even if the MLE is not totally efficient.
  - There $P((\alpha/2) \leq \sqrt{nI(\hat{\theta})}(\hat{\theta} −\theta_0) \leq z(\alpha/2)) \approx 1 - \alpha$
  - This gives us $\hat{\theta} +- z(\alpha/2)\frac{1}{\sqrt{nI(\hat{\theta})}} $
  - Similarly, $Var(\hat{\theta}) \approx \frac{1}{El’(\theta_0)^2} = -\frac{1}{El’’(\theta_0)}$ and the MLE is approximately normally distributed
- Bootstrap
  - Assume we know the distribution of $\delta = \hat{\theta} - \theta_0$. Denote $\alpha/2$ and $1-\alpha/2$ as quantiles of this distribution by $\underline{\delta}, \, \bar{\delta}$
  - $P(\underline{\delta}\leq \hat{\theta} - \theta_0 \leq \bar{\delta}) = 1 - \alpha$  so $P(\hat{\theta} - \bar{\delta}\leq  \theta_0 \leq  \hat{\theta} - \underline{\delta}) = 1 - \alpha$ 
  - If $\theta_0$ were known, this distribution could be approximated  arbitrarily well by simulation. Since $\theta_0$ is not known, the bootstrap principle suggests using $\hat{\theta}$ in its place: Generate many, many samples (say, B in all) from a distribution with value $\hat{\theta}$; and for each sample construct an estimate of $\theta$.

##### Bayesian Estimation

- Unknown parameter is treated as an RV with a prior distribution $f_\Theta(\theta)$ what we know about the parameter before observing data X, eg. we might think a failure rate is modeled by an exponential process. When one treats the parameter as a random variable, the name “prior distribution” is merely another name for the marginal distribution of the parameter. Assume $\Theta$ is a continuous RV.
- Therefore we can get the joint distribution of X and $\Theta$ by $f_{X,\Theta}(x, \theta) = f_{X|\Theta}(x|\theta)f_\Theta(\theta)$ When one treats the parameter as a random variable, the name “posterior distribution” is merely another name for the conditional distribution of the parameter given the data. For many random variables iid (ie data), $f_n(x_1,...,x_n|\theta) = f(x_1|\theta) ... f(x_n|\theta).$
- And we can use Bayes rule to get the posterior proportional to likelihood times prior : $f_{\Theta|X}(\theta|x) \propto f_{X|\Theta}(x|\theta)f_\Theta(\theta)$. Why? The $\int f_{X|\Theta}(x|\theta)f_\Theta(\theta) \, d\theta$ term (ie. the marginal of X) is simply the integral of the numerator over all possible values of θ. Although the value of this integral depends on the observed values $x_1,...,x_n$, it does not depend on $\theta$ and it may be treated as a constant when $\frac{f_{X,\Theta}}{f_x(x)}$  is regarded as a p.d.f. of $\theta$. The appropriate constant factor that will establish the equality of the two sides Bayes rule can be determined at any time by using the fact that  $\int_{\Omega}f(\theta|x) d\theta = 1$, because $f(\theta|x)$ is a p.d.f. of $\theta$. Often we can do this without integration by recognizing the posterior as a known probability distribution missing a constant.
- Essentially $f_{\Theta|X}(\theta|x) = \frac{f_{X,\Theta}}{f_x(x)} = \frac{f_{X|\Theta}(x|\theta)f_\Theta(\theta)}{\int f_{X|\Theta}(x|\theta)f_\Theta(\theta) \, d\theta}$. The marginal of X is the joint pdf integrated over all values of $\theta$.
- Steps to find the posterior distribution
  - Find the prior distribution
  - Find $f(x|\theta)$, ie. the distribution of the data in terms of the parameter.
  - Calculate $f(x|\theta)f(\theta)$
  - Try to figure out constant from a recognized probability distribution. If must integrate of the denominator of Bayes, ie $\int f_{X|\Theta}(x|\theta)f_\Theta(\theta) \, d\theta$
  - Note: Now, consider this an important trick that is used time and again in Bayesian calculations: the denominator is a constant that makes the expression integrate to 1. We can deduce from the form of the numerator that the ratio must be a gamma density
- We can also calculate sequentially using one data point at a time to obtain the same posterior, using the posterior from one observation as the prior for the next. For improper priors we use the posterior is proportional to the likelihood and ensure the likelihood integrates to 1
- Steps for finding an estimator $\hat{\theta}$
  - It could be the mean, median or mode of the posterior distribution.
  - The posterior mean minimizes the posterior mean squared error $\mathbb{E}_{\Theta | X}\left((\hat{\theta}-\Theta)^{2} | x\right)$
  - The posterior median minimizes the posterior mean absolute error $\mathbb{E}_{\Theta | X}\left(|\hat{\theta}-\Theta| | x\right)$
- Bayes estimators are consistent

##### Cramer-Rao Lower Bounds / Efficiency

- Given two estimates $\hat{\theta}, \bar{\theta}$ of the same parameter, the efficiency of $\hat{\theta}$ relative to $\bar{\theta}$ is defined to be $eff(\hat{\theta, \bar{\theta}}) = \frac{Var(\bar{\theta})}{Var(\hat{\theta})}$
- If eff is smaller than 1, then theta-hat has a larger variance than theta-bar. Most meaningful when the estimates have the same bias. if Var(estimator) = $\frac{c}{n}$, then the efficiency is the ratio os sample sizes necessary to obtain the same variance for both estimators.
- Cramer Rao inequality: $X_1,X_2,...,X_n$ iid with density function $f(x|\theta)$. Let $T = t(X_1,X_2,...,X_n)$ be an unbiased estimate of $\theta$. Then $Var(T) \geq \frac{1}{nI(\theta)}$. If it is an equality, then T is an efficient estimator. 
- More general version: T statistic with finite variance, $m(\theta) = E_\theta(T)$ $\operatorname{Var}_{\theta}(T) \geq \frac{\left[m^{\prime}(\theta)\right]^{2}}{n I(\theta)}$. When T goes to theta in expectation, then the numerator is the derivative of theta = 1.
- The variance of an unbiased estimator of $\theta$ cannot be smaller than the reciprocal of the Fisher information in the sample.

- Theorem A gives a lower bound on the variance of any unbiased estimate. An  unbiased estimate whose variance achieves this lower bound is said to be efficient.  Since the asymptotic variance of a maximum likelihood estimate is equal to the  lower bound, maximum likelihood estimates are said to be asymptotically efficient. 
- If T is an efficient estimator of $m(\theta)$, then among all unbiased estimators of $m(\theta)$, T will have the smallest variance for every possible value of $\theta$.


##### Sufficiency

- Given the initial conditions for Cramer Rao above. 
- Imagine one statistician who can observe the data and one who only gets a statistic about the data. Is there statistic such that observing the data provides no additional information. A sufficient statistic is sufficient for being able to compute the likelihood function, and hence it is sufficient for performing any inference that depends on the data only through the likelihood function. M.L.E.’s and anything based on posterior distributions depend on the data only through the likelihood function.
- The concept of sufficiency arises as an attempt  to answer the following question: Is there a statistic, a function $T (X_1,X_2,...,X_n)$, that  contains all the information in the sample about $\theta$? If so, a reduction of the original  data to this statistic without loss of information is possible. For example, in series of Bernoulli trials, total number of successes contains all of the information about p that exists in the sample.
- Sufficiency: A statistic $T (X_1,X_2,...,X_n)$ is said to be sufficient for $\theta$ if the conditional distribution of $X_1,X_2,...,X_n$ given $T = t$, does not depend on $\theta$ for any value  of t.
- In other words, given the value of T , which is called a sufficient statistic, we can  gain no more knowledge about $\theta$ from knowing more about the probability distribution  of $X_1,X_2,...,X_n$
- Factorization: A necessary and sufficient condition for $T (X_1,X_2,...,X_n)$ to be sufficient for a  parameter  $\theta$ is that the joint probability function (density function or frequency  function) factors in the form: $f(x_1,...,x_n|\theta) = g[T(x_1,...,x_n), \theta]h(x_1,...,x_n)$. u depends on x but not theta. g depends on theta but only depends on x through the statistic. 

###### Rao Blackwell Theorem

- Let $\hat{\theta}$ be an estimator of $\theta$ with existing expectation for all theta. If T is sufficient for theta and $\bar{\theta} = E(\hat{\theta}|T)$. Then for all theta: $E(\bar{\theta} - \theta) \leq E(\hat{\theta} - \theta)^2$
- Rationale for basing estimators on sufficient stats if they exist. Conditioning on T is sure to give a function of the data, not a function of the true parameter theta that cannot be observed.