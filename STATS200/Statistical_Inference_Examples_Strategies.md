---
title: Statistical Inference - Examples and Strategies
date: 20191029
author: Spencer Braun
---

**Change of Variables**

* Jacobians:
  * for $(x,y) \rightarrow (r, \theta)$, $x = rcos(\theta), y = rsin(\theta)$
  * $J=\left|\begin{array}{cc}{\cos \theta} & {\sin \theta} \\ {-r \sin \theta} & {r \cos \theta}\end{array}\right|=r$ 
  * Jacobian takes partials of the old variables wrt the new variables
  * Jacobian is positive and takes the absolute value of the calculation. Can take the jacobian for the other direction and then invert for direction given in problem.
* CDF approach
  * start from CDF, plug in changed variable, and differentiate
  * $Y=a X+b$
  * $P\left(X \leq \frac{y-b}{a}\right) = F_{X}\left(\frac{y-b}{a}\right)$
  * $\frac{1}{a} f_{X}\left(\frac{y-b}{a}\right)$
* $F_{T_{1}+T_{2}}(t)=\mathbb{P}\left(T_{1}+T_{2} \leq t\right)$

**Constructing Confidence Intervals**

* Goal here is to get the population parameter as a pivot and the inequality includes only statistics and known quantities. If you normalize, set between z-scores, then do algebra to extricate the population parameter from any other terms.

* A pivot is a statistic whose distribution does not depend on any unknown parameter, even though the pivot statistic itself can have unknown parameters in it. 

* Normalize variable

  * population $\frac{X - \mu}{\sigma}$
  * sample $\frac{\theta - \bar{\theta}}{SE}$

* population parameter = estimator $\pm SE \times CV$

* $P(|Z| \leq CV) = P(-CV \leq Z \leq CV)$

  | 1 - $\alpha$ | Z-Table CV |
  | ------------ | ---------- |
  | 0.90         | 1.29       |
  | 0.95         | 1.65       |
  | 0.975        | 1.96       |
  | 0.99         | 2.33       |
  | 0.995        | 2.58       |

* | Sig  | 1 tail | 2 tail |
  | ---- | ------ | ------ |
  | 0.90 | 1.29   | 1.65   |
  | 0.95 | 1.65   | 1.96   |
  | 0.99 | 2.33   | 2.58   |

**MOM Estimate**

* Produce moments until moment equations = number of parameters to estimate
* If given the mean and variance, can solve for the 2nd moment: $E(X^2) = Var(X) + E(X)^2$
* Solve for parameters in term of moments
* Plug in sampling moments for moments, eg. $\hat{\mu}_1 = E(X) = \bar{X}$, $\hat{\mu}_2 = E(X^2) = \frac{1}{n}\sum_1^n X^2$

**MLE Estimate**

* Write down likelihood $lik(\theta) = f(x|\theta)$. If estimator for a single data point X, then it just the density function. **For multiple X’s, take the product of their density functions.**
* Take the log of the likelihood function $l(\theta) = log(f(x|\theta))$ and simplify. May be simplest to have sum of logs, then slowly distribute the summation.
* Take the first derivative and set it equal to zero. $l'(\theta) = 0$
* Solve for parameter $\theta$ in terms of X’s to get MLE estimate $\hat{\theta}$
* Check that it is a maximum using 2nd derivative test, but plugging in our MLE for $\theta$: $l’’(\hat{\theta}) < 0$
* Check bounds if multiple roots 
* To find asymptotic variance
  * calculate $I(\theta)$ and the AV = $\frac{1}{nI(\theta)}$
  * If I just for single variable, then multiple by n variables. If n accounted for in expectation, do not multiply by another n, since $nI(\theta) = I_n(\theta)$
  * To compare to MLE, take $Var(\hat{\theta}) = Var(\theta(X))$ and use the distribution of X to find the variance of the estimator.
* Showing it is unbiased - if MLE is $\bar{X}$, place back into its summation for and apply linearity of expectation to E(X)

**Bayesian Estimation**

* Given a likelihood and a prior
* Multiply prior and likelihood densities. If multiple X’s, will be a product of this new density
* Simplify the combined densities to find a recognizable density function and define that densities parameters. Easier to ignore constants in terms of the parameter to estimate and just try to see a known distribution. If conjugate is given as a prior, the goal should be known beforehand
* To find mean or median, rely on the theoretical density recognized in previous step. 

**Neyman Pearson Testing**

* $\begin{aligned} \alpha &=P(\text { type I error })=P\left(\text { rejecting } H_{0} \text { when } H_{0} \text { is true }\right) \\ &=P \text { (value of test statistic is in RR when } H_{0} \text { is true) } \\ &=P(Y \leq 2 \text { when } p=.5) \end{aligned}$
* B = P(type II error) = P(accepting H0 when Ha is true) = P(value of the test statistic is not in RR when Ha is true)
* Z = (estimator for parameter - value for parameter under H0) / Standard error for estimator
* With CI we solve for population parameter and give bounds for its value from the estimator. With HT we guess the population value and determine if the estimator from the data gives us evidence to agree or disagree.
* Small sample testing - we use the t stat instead of normal curves
* Test statistic for 2 sample means: $T=\frac{\bar{Y}_{1}-\bar{Y}_{2}-D_{0}}{S_{p} \sqrt{\frac{1}{n_{1}}+\frac{1}{n_{2}}}}$ for $S_{p}=\sqrt{\frac{\left(n_{1}-1\right) S_{1}^{2}+\left(n_{2}-1\right) S_{2}^{2}}{n_{1}+n_{2}-2}}$
* Tests with variances under normality assumptions:
  * $\chi^{2}=\frac{(n-1) S^{2}}{\sigma_{0}^{2}} \sim \chi^2_{(n-1)}$
  * The procedure is the same, but the chi square distribution is not symmetric.
  * Solve for $S^2$, plug in null guess for $\sigma^2_0$
  * Can test two variances are equal with a ratio of variances by noting they are both chi square and using an F distribution: $F=\frac{\left(n_{1}-1\right) S_{1}^{2}}{\sigma_{1}^{2}\left(n_{1}-1\right)} / \frac{\left(n_{2}-1\right) S_{2}^{2}}{\sigma_{2}^{2}\left(n_{2}-1\right)}=\frac{S_{1}^{2} \sigma_{2}^{2}}{S_{2}^{2} \sigma_{1}^{2}}$. If H0 is $\sigma_2^2 = \sigma^2_1$, then test stat = $\frac{S^2_1}{S^2_2}$

**LRT for Simple Hypotheses**

* Take ratio of density with null parameter over density with alternative parameter: $\frac{f\left(x | \sigma_{0}\right)}{f\left(x | \sigma_{1}\right)}$
* Simplify to get a single expression in terms of both parameters. Set LRT less than constant k and solve for x or function of x.
* Determine how movements in X affect the LRT - ie. if X increases does the LRT increase or decrease. Null is rejected for small values of LRT
* The form of the test statistic depends on what comes out of the LRT, but want to isolate the variable and consolidate constants into the compared critical value.
* Define rejection region / test for level $\alpha$
  * Define distribution of X under null hypothesis 
  * Determine the CV needed for given signficance level and perform hypothesis test / CI just like we would for any Neyman Pearson test
  * UMP if likelihood test did not depend on any values from the alternative hypothesis
  * Example of LRT test: $.05=P(Y \text { in RR when } \theta=2)=P\left(Y<k^{*} \text { when } \theta=2\right)=\int_{0}^{k} 2 y d y=\left(k^{*}\right)^{2}.$ Then $k^* = \sqrt{0.05}$ and RR is given by $P(Y < \sqrt{0.05})$
  * Power is $P(Y < \sqrt{0.05})$ under the alternative distribution

**GLRT**

* $\Lambda=\frac{\max (\operatorname{lik}(p))_{p \in \omega_{0}}}{\max (\operatorname{lik}(p))_{p \in \Omega}}$
* Plug in the parameter values hypothesized by H0 into the numerator. 
* Find the MLE of the distribution, plug that into the denominator
* If there is a nuissance parameter in H0, it may have to be estimated, Usually by MLE for that parameter.
* Determine how movements in X affect the LRT - ie. if X increases does the GLRT increase or decrease. Null is rejected for small values of GLRT. This will determine which direction we set up our probability test.
* To find the CV or signficance for given values, use the distribution under H0. If we have the exact distribution, set up P(X < CV ) = alpha (or P(X > CV ) depending on results from previous step).
* We often use the approximation $-2log(\Lambda) \approx \chi^2_d$ under H0. Then RR $-2 \ln (\lambda)>\chi_{\alpha}^{2}, \quad \text { where } \chi_{\alpha}^{2} \text { is based on } r_{0}-r \text { df. }$
  * Used for 2 sided tests
  * Find chi square quantile needed for significance level $\alpha$
  * Calculate $-2log(\Lambda)$ and reject if $-2log(\Lambda) > \chi^2$ quantile found
* Instead can use a normal approximation. Set up probability as before, then normalize, use z-table.

**Power of a Test**

* We have set a normalized variable against a CV to determine where we reject for null hypothesized $\mu = \mu_0$, eg. $P(\frac{(\bar{X} - \mu_0)}{s / \sqrt{n}} < 1.96) \rightarrow P(\bar{X} < 1.96\frac{s}{\sqrt{n}} + \mu_0)$. Call $CV_0 = 1.96\frac{s}{\sqrt{n}} + \mu_0$
* The power are the tails of the H1 distribution greater and less than the critical values calculated above for rejecting the null hypothesis. Intuitively, this is the probability that H1 would take on values in the H0 rejection region when it is the truth. Only when H1 takes on values in the RR will we reject H0, so the power is equal to how likely it is to take those values.
* Power of the test given $\mu = \mu_1$: $P(reject \,H_0|\mu=\mu_1) \rightarrow P(\bar{X} < CV_0 | \mu=\mu_1) \rightarrow P(\frac{(\bar{X} - \mu_1)}{s / \sqrt{n}} < \frac{(CV_0 - \mu_1)}{s / \sqrt{n}}) \rightarrow P(Z < CV_z)$
* Find the likelihood for the normalized distribution for those areas (ie, look up on the z table). For two sided, will have two values to look up.
* Draw the graph - then decide which areas need to be included. Set up probability and standardize.
* By finding B
  * Find rejection region by finding alpha’s critical value as usual. End with $P(\bar{X} >  CV) = \alpha$
  * We now want to find the probability that $\bar{X} < CV$ using the H1 distribution
  * $P(\bar{X} < CV) \rightarrow normalize \rightarrow P(\frac{\bar{X} - \mu1}{\sigma/\sqrt{n}} < \frac{CV - \mu1}{\sigma/\sqrt{n}}) = P(Z < zstat) = B$
  * 1- B is power

**Chi Square Test**

* Consider a multinomial experiment. n distinct, independent trials, k distinct categories. The probability of ending in a certain category is $p_i, \, i \in 1,2,...,k$. $p_{1}+p_{2}+p_{3}+\cdots+p_{k}=1$. For $n_i$ equal to the number of trials that fall into bucket i, $n_{1}+n_{2}+n_{3}+\dots+n_{k}=n$. 
* $X^{2}=\sum_{i=1}^{k} \frac{\left[n_{i}-E\left(n_{i}\right)\right]^{2}}{E\left(n_{i}\right)}=\sum_{i=1}^{k} \frac{\left[n_{i}-n p_{i}\right]^{2}}{n p_{i}}$
* Each n has a binomial marginal. Expectation of a binomial is the expected outcome. 
* Because large differences between the observed and expected cell counts contradict the null hypothesis, we will reject the null hypothesis when X2 is large and employ an upper-tailed statistical test.
* The appropriate number of degrees of freedom will equal the number of cells, k, less 1 df for each independent linear restriction placed on the cell probabilities (and we always have at least 1, the one mentioned above).
* With chosen alpha, find chi square critical value. Compute Chi Square test statistic and compare. If larger, reject $O \sim E$. 
* Can plug in MLE if unknown parameter is in equation.

**Other**

* Integration bounds when support depends on other variable - 
  * Marginal densities can include the variable in the bound. 
  * Expectation take absolute bounds. 
  * Conditional expectation E(Y | X = x) - the bounds should include x, since x is now a specified value
  * Double integrals the inner integral bounds should depend on a variable, outer are absolute.
* Specify support for derived distribution, often have division by a variable
* Independence can be shown by taking $\frac{f_{X,Y}}{f_Y}$ and noting it does not depend on y 
* Density function of expectation $E(X|Y)$ -> change of variables with $\mathbb{E}(X | Y=y)$, plug into marginal density function of Y. Why? This is a function of y, g(Y).
* When confronted with convergence, perhaps directly plug into Chebyshev, CLT, WLLN, other theorems dealing with convergence in probability.
* When p is a proportion of a binary outcome in a finite population size: $Var(\hat{p}) = s_{\hat{p}}^{2}=\frac{\hat{p}(1-\hat{p})}{n}$

**MGF Derivations**

Poisson

* $\boldsymbol{M}_{X}(t)=\mathrm{E}\left(e^{t X}\right) = \sum_{k=0}^{\infty} \frac{\lambda^{k} }{k !} e^{-\lambda}e^{t k}$
* $= e^{-\lambda}\sum_{n=0}^{\infty} \frac{(\lambda e^{t})^{k} }{k !} $   power series expansion
* $= e^{-\lambda} e^{\lambda e^{t}} = e^{\lambda(e^t -1)}$

Normal

* For standard normal $M(t) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^\infty e^{tx} e^{-\frac{x^2}{2}} dx$
* Using $\frac{x^{2}}{2}-t x = \frac{1}{2}(x-t)^{2}-\frac{t^{2}}{2}$
* $M(t)=\frac{e^{t^{2} / 2}}{\sqrt{2 \pi}} \int_{-\infty}^{\infty} e^{-(x-t)^{2} / 2} d x $ substituting $u = x-t$
* $= e^{t^2/2}$
* For $N(\mu, \sigma)$, $M(t) = e^{\mu t}M(\sigma t) = e^{\mu t}e^{\sigma^2 t^2 /2}$

Geometric

* $M(t) = \sum_{k=0}^\infty e^{tk} p(1-p)^{k-1}$
* $=p e^t \sum e^{t(k-1)}(1-p)^{k-1}$
* $=p e^t \sum (e^t(1-p))^{k-1}$ and since $(1-p)e^t < 1$, use geometric series formula
* $= \frac{pe^t}{1-e^t(1-p)}$

Exponential

* $M(t) = \int_{-\infty}^\infty e^{tx} \lambda e^{-\lambda x} dx$
* $=\lambda \int_0^\infty e^{x(t-\lambda)} dx$
* $=\frac{\lambda}{t-\lambda}e^{x(t-\lambda)} \Big|_0^\infty = \frac{\lambda}{\lambda -t}$

Binomial

* $M(t) = \sum_{k=0}^\infty {n \choose k} e^{tk} p^k (1-p)^{n-k}$
* $ = \sum_{k=0}^\infty {n \choose k} (e^t p)^k (1-p)^{n-k}$ apply binomial theorem
* $= (1-p + pe^t)^n$

Gamma

* $M(t)=\int_{0}^{\infty} e^{t x} \frac{\lambda^{\alpha}}{\Gamma(\alpha)} x^{\alpha-1} e^{-\lambda x} d x$
* $=\frac{\lambda^{\alpha}}{\Gamma(\alpha)} \int_{0}^{\infty} x^{\alpha-1} e^{x(t-\lambda)} d x$, where $\int_{0}^{\infty} x^{\alpha-1} e^{x(t-\lambda)} d x$ is a gamma density $g(\alpha, \lambda -t)$
* $M(t)=\frac{\lambda^{\alpha}}{\Gamma(\alpha)}\left(\frac{\Gamma(\alpha)}{(\lambda-t)^{\alpha}}\right)=\left(\frac{\lambda}{\lambda-t}\right)^{\alpha}$

Uniform

* $M(t) = \int_a^b e^{tx} \frac{1}{b-a} dx$
* $=\frac{1}{b-a} \frac{1}{t}e^{tx}\Big|_a^b$
* $ = \begin{cases}\frac{e^{bt} - e^{at}}{t(b-a)} & t>0 \\ 1 & t=0 \end{cases}$

**Good Problems to Review**

* 3.24 - Bayes rule mixing discrete and continuous
* 6.5 - 6.9 - Weigh costs and benefits of variable manipulation vs density manipulation
* 8.16 c,d - question about the steps taken
* 8.60 - MLE 
* 9.17, 9.18 LRT
* 9.24 GLRT