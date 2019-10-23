---
title: Statistical Inference - Hypothesis Testing
author: Spencer Braun
date: 20191016
---

[TOC]

### Chapter 9 - Testing Hypotheses and Goodness of Fit

* Likelihood Ratio - For two distributions, the probability of observing an event from one over the other.
  * Bayesian approach: $\frac{P\left(H_{0} | x\right)}{P\left(H_{1} | x\right)}=\frac{P\left(H_{0}\right)}{P\left(H_{1}\right)} \frac{P\left(x | H_{0}\right)}{P\left(x | H_{1}\right)}$
  * The product of the ratio of prior probabilities and the likelihood ratio. Thus, the  evidence provided by the data is contained in the likelihood ratio, which is multiplied  by the ratio of prior probabilities to produce the ratio of posterior probabilities.

##### Neyman-Pearson Paradigm

* Null hypothesis $H_0$ and alternative hypothesis $H_1$
  * Type I error: rejecting null when it is true. The probability of type I error is the significance level of the test $\alpha$ - jury returns guilty when innocent
  * Type II error: accepting the null hypothesis when false, denoted by $\Beta$. Jury says innocent when guilty
    * Inherent trade off between Type I error and Type II - Type I maximized when Type II minimized and vice versa
  * Power: the probability that the null hypothesis is rejected when false, $1-\Beta$
  * Test statistic, statistic we are using as a test of the hypothesis. A function of sample we are going to test - eg. LRT
  * Null distribution: the probability distribution of the test statistic when the null hypothesis is true

1. Determine hypotheses and set an alpha for the test
2. Either compute critical value from PDF of x and alpha (don’t need x) or compute p-value from x and its PDF (don’t need alpha)
3. Determine if we reject null or not

* Suppose that $H_0$ and $H_1$ are simple hypotheses and that the test that rejects $H_0$  whenever the likelihood ratio is less than c and significance level $\alpha$. Then any other test for which the significance level is less than or equal to $\alpha$ has power less than or equal to that of the likelihood ratio test.
* For simple hypothesis: We write down the likelihood ratio and observe that small values of it correspond in a  one-to-one manner with extreme values of a test statistic, in this case X. Knowing the null distribution of the test statistic makes it possible to choose a critical level that  produces a desired significance level $\alpha$.
* p-value is the smallest alpha for which we reject the null hypothesis - critical value < test stat equivalent to p-value < alpha
* ![hypothesis_test_regions](/Users/spencerbraun/Documents/Notes/Stanford/STATS200/hypothesis_test_regions.png)
* If a hypothesis does not completely specify the probability distribution, the hypothesis is called a composite hypothesis. This means a hypothesis like this is Poisson distributed or not is composite, because the null hypothesis needs to be a specific distribution with specified parameter to be simple. It is convention to choose the simpler hypothesis to be the null.
*  If the alternative $H_1$ is composite, a test that is most powerful for every simple alternative  in $H_1$ is said to be uniformly most powerful. 
* In typical composite situations, there is no uniformly most powerful test. The alternatives $H_1 : \mu < \mu_0$ and $H_1 : \mu> \mu_0$ are called one-sided alternatives. The  alternative $H_1 : \mu = \mu_0$ is a two-sided alternative. 
* Confidence Intervals: $\mu_0$ lies in the confidence interval for $\mu$ if and only if the hypothesis test accepts. In  other words, the confidence interval consists precisely of all those values of $\mu_0$ for  which the null hypothesis $H_0: \mu= \mu_0$ is accepted.
* Suppose that for every value $\theta_0  \in \Theta$ there is a test at level $\alpha$ of the hypothesis  $H_0: \theta=\theta_0$. Denote the acceptance region of the test by $A(\theta_0)$. Then the set  $C(X) = {\theta: X \in A(\theta)}$  is a $100(1−\alpha)\%$ confidence region for $\theta$. Basically if a value for theta lies in the confidence region, the hypothesis test would be accepted for that value.

##### Generalized Likelihood Ratio Tests

* $\Lambda=\frac{\max _{\theta \in \omega_0 }[\operatorname{lik}(\theta)]}{\max _{\theta \in \Omega}[\operatorname{lik}(\theta)]}$ - Testing the parameter space for observations.
* For $H_0: \mu = \mu_0,\,H_1:\mu \neq \mu_0$, the numerator of the likelihood ratio is the density function at point $\mu_0$ and the denominator we plug in the mle of the parameter $\mu$, (so here for example, $\bar{X}$) 
* Under smoothness conditions on the probability density or frequency functions  involved, the null distribution of $−2 log \Lambda$ tends to a chi-square distribution  with degrees of freedom equal to  $dim\,\Omega−dim\,ω_0$ as the sample size tends to  infinity.
* Generally - set up the distributions, compose the likelihood ratio, often take the log of both sides, try to determine how the likelihood changes for different observed values.

* $-2 \log \Lambda \approx \sum_{i=1}^{m} \frac{\left[x_{i}-n p_{i}(\hat{\theta})\right]^{2}}{n p_{i}(\hat{\theta})} = X^2$ : RHS is Pearson’s test statistic for goodness of fit. Degrees of freedom (df): # of free parameters (think: # of unknown parameters) 
* Pearson Chi-Square Test $X^2 = \sum\frac{(Observed - Expected)^2}{Expected}$
  * $\chi_{\text {stat }}^{2}:=\sum_{1}^{n} \frac{\left(O_{i}-E_{i}\right)^{2}}{E_{i}} \sim \chi_{\mathrm{d} f:=n-s-1}^{2}$
  * Goodness-of-fit: whether eCDF differs from *any* theoretical CDF 
  * 1) build test statistic for n bins - Oi is the observed count in each bin and Ei is the expected count in each bin
  * 2) null hypoth is O ~ E, so build test statistic, find p-value, decide whether to accept or reject

##### Poisson Dispersion Test

* If one  has a specific alternative hypothesis in mind, better power can usually be obtained  by testing against that alternative rather than against a more general alternative.
* The two key assumptions underlying the Poisson distribution are that the rate  is constant and that the counts in one interval of time or space are independent of  the counts in disjoint intervals. These conditions are often not met.
* Given counts $x_1,..., x_n$, we consider testing the null hypothesis that the counts  are Poisson with the common parameter $\lambda$ versus the alternative hypothesis that  they are Poisson but have different rates, $\lambda1,...,\lambda_n$.
* $\begin{aligned} \Lambda &=\frac{\prod_{i=1}^{n} \hat{\lambda}^{x_{i}} e^{-\hat{\lambda}} / x_{i} !}{\prod_{i=1}^{n} \tilde{\lambda}_{i}^{x_{i}} e^{-\bar{\lambda}_{i}} / x_{i} !} =\prod_{i=1}^{n}\left(\frac{\bar{x}}{x_{i}}\right)^{x_{i}} e^{x_{i}-\bar{x}} \end{aligned}$
* $-2 \log \Lambda \approx \frac{1}{\bar{x}} \sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}$ using Taylor approximation. 
* This equation is roughly the ratio of  n times the estimated variance to the estimated mean. For the Poisson distribution, the  variance equals the mean; for the types of alternatives discussed above, the variance is typically greater than the mean. For this reason the test is often called the Poisson dispersion test.

##### Hanging Rootograms

* Hanging rootograms are a  graphical display of the differences between observed and fitted values in histograms. 
* Suppose we estimate $\bar{x} \rightarrow \mu, \, \hat{\sigma} \rightarrow \sigma$, the then probability that an observation falls in an interval between $x_{j-1}, x_j$ is $\hat{p}_{j}=\Phi\left(\frac{x_{j}-\bar{x}}{\hat{\sigma}}\right)-\Phi\left(\frac{x_{j-1}-\bar{x}}{\hat{\sigma}}\right)$. For sample size n, the predicted count in the jth interval is $\hat{n}_j = n\hat{p}_j$, which we then compare to the observed counts
* The hanging histogram is then the difference between the fitted frequency of the bins and the observed. Expect larger fluctuations in the center than in the tails since variance across buckets is not constant - can use a variance-stabilizing transformation. Often use $f(x) = \sqrt{x}$ so get a hanging rootogram showing $\sqrt{n_j} - \sqrt{\hat{n}_j}$
* Generally, deviations in the center  have been down-weighted and those in the tails emphasized by the transformation. 
* Hanging chi-gram: plots $\frac{n_j - \hat{n}_j}{\sqrt{\hat{n}_j}}$

##### Probability Plots

* Useful graphical tool for qualitatively assessing the  fit of data to a theoretical distribution
* Plotting the ordered observations against expected values (find order statistic of data and plot against quantiles of a known distribution)
* Probability integral transform: $Y = F_X(X)$ to get uniform distribution. Then can plot against uniform quantiles.

#####  Tests for Normality

* A goodness-of-fit test can be based on the coefficient of skewness: $b_{1}=\frac{\frac{1}{n} \sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)^{3}}{s^{3}}$ , s is sample SD. Negative skew is left-sided: left tail is longer than right tail.
* The test rejects for large values of |b1|
* Coefficient of kurtosis: $b_{2}=\frac{\frac{1}{n} \sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)^{4}}{s^{4}}$ . Normal has kurtosis of 3, so could have null hypothesis kurtosis = 3 and reject for large values of $|b_2 - 3|$.
* A goodness-of-fit test may also be based on the linearity of the probability  plot, as measured by the correlation coefficient, r, of the x and y coordinates of  the points of the probability plot. The test rejects for small values of r.
* Kolmogorov Smirnov Test - $D_{n}=\max _{x}\left|F_{n}(x)-F(x)\right|$ - largest vertical distance between the eCDF and the CDF.

### Chapter 10 - Summarizing Data

##### CDF Methods

* empirical CDF: $F_{n}(x)=\frac{1}{n}\left(\# x_{i} \leq x\right)=\frac{1}{n} \sum_{i=1}^{n} I_{(-\infty, x]}\left(X_{i}\right)$ for a batch of numbers X, then just a count of numbers less than a specified value over the total batch size. Note indicators are Bernoullis: $I_{(-\infty, x]}\left(X_{i}\right)=\left\{\begin{array}{ll}{1,} & {\text { with probability } F(x)} \\ {0,} & {\text { with probability } 1-F(x)}\end{array}\right.$
  * $n F_{n}(x)$ is a binomial RV - n trials with F(x) probability of success
  * $\begin{aligned} E\left[F_{n}(x)\right] &=F(x) \\ \operatorname{Var}\left[F_{n}(x)\right] &=\frac{1}{n} F(x)[1-F(x)] \end{aligned}$
* Survival Function: $S(t)=P(T>t)=1-F(t)$
  * Simply a reversal of the CDF for data consist of time until death or failure.
* Hazard Function: as the instantaneous death rate for individuals who  have survived up to a given time. $h(t)=\frac{f(t)}{1-F(t)}$

##### QQ Plots

* Pth quantile: $F(x) = p$ or $x_p = F^{-1}(p)$
* Plot the quantiles of one distribution against another
* Additive: if $y_{p}=x_{p}+h$, then for the quantiles $G(y)=F(y-h)$
* Multiplicative: if $y_{p}=c x_{p}$, then for quantiles $G(y)=F(y / c)$

##### Histograms, Density Curves, Stem / Leaf plots

* Smooth probability density estimate: using w, smooth weight function integrating to 1 and centered at 0, 
* Kernel probability density estimate: $f_{h}(x)=\frac{1}{n} \sum_{i=1}^{n} w_{h}\left(x-X_{i}\right)$ - superposition of hills centered on the observations. If w(x) is standard normal w(x - Xi) is normal with mean Xi and SD h. The parameter h, bandwidth, controls the smoothness and is the bin width of the histogram. With histograms and density estimates, we lose information and cannot reconstruct the data.
*  Stem and leaf plots - retain numerical information while showing shape. 

##### Measures of Location

* Measure of the center of batch of numbers
* Arithmetic mean - sum over the count
* robust measures  - insensitive to outliers
* When the data are a sample from a continuous probability law, the sample median can be viewed as an estimate of the population median. The distribution of the number of observations greater than the median is binomial  with n trials and probability 1/2 of success on each trial.
* Trimmed mean - mean. The 100α%  trimmed mean is easy to calculate: Order the data, discard the lowest 100α% and the  highest 100α%, and take the arithmetic mean of the remaining data: $\bar{x}_{\alpha}=\frac{x_{([n \alpha]+1)}+\cdots+x_{(n-[n \alpha])}}{n-2[n \alpha]}$
* M Estimates - minimizers of $\sum_{i=1}^{n} \Psi\left(\frac{X_{i}-\mu}{\sigma}\right)$. where the weight function is a compromise between the weight functions for the  mean and the median.
* Estimating variability of location estimates by bootstrap:
  * n. Suppose we denote the location estimate asθˆ; it is important to keep in  mind that $\hat{\theta}$ is a function of the random variables $X_1, X_2,..., X_n$ and hence has a probability distribution, its sampling distribution, which is determined by n and F. We don’t know F and $\hat{\theta}$ may be complicated.
  * We generate many samples from F (if we knew it) and calculate the value of $\hat{\theta}$, then could find measures on the samples like SD. Use empirical CDF instead as an approximation of F. Then the SD is estimated as: $s_{\hat{\theta}}=\sqrt{\frac{1}{B} \sum_{i=1}^{B}\left(\theta_{i}^{*}-\bar{\theta}^{*}\right)^{2}}$

##### Measures of Dispersion

* sample standard deviation: $s^{2}=\frac{1}{n-1} \sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)^{2}$ (use n-1 as divisor to make s^2 unbiased estimate of population variance)
  * If sample from standard normal, $\frac{(n-1) s^{2}}{\sigma^{2}} \sim \chi_{n-1}^{2}$
* Median absolute deviation from the median (MAD): the data  are $x_1,..., x_n$ with median $\tilde{x}$, the MAD is defined to be the median of the numbers  $|x_i − \tilde{x}|$.

##### Boxplot

* The range lines - A vertical line is drawn up from the upper quartile to the most extreme data point that is within a distance of 1.5 (IQR) of the upper quartile. A similarly defined vertical line is drawn down from the lower quartile. Short horizontal lines are  added to mark the ends of these vertical lines. 
