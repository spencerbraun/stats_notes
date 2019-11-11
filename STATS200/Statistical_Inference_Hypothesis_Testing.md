---
title: Statistical Inference - Hypothesis Testing
author: Spencer Braun
date: 20191016
---

[TOC]

### Chapter 9 - Testing Hypotheses and Goodness of Fit

##### Neyman-Pearson Paradigm

* Null hypothesis $H_0$ and alternative hypothesis $H_A$
  * **Type I error**: rejecting null when it is true, FP. The probability of type I error is the **significance level of the test $\alpha$** - jury returns guilty when innocent. $\alpha = P(T(X_1,...,X_n) > t_0 | H_0)$
  * **Type II error**: accepting the null hypothesis when false, FN. Denoted by $\Beta$. Jury says innocent when guilty
    * $\beta = P(T(X_1,...,X_n) \leq t_0 | H_A)$
    * Inherent trade off between Type I error and Type II - Type I maximized when Type II minimized and vice versa
  * p-value: $p = P(T(X_1,...,X_n) \geq T(x_1,...,x_n) | H_0)$ The chance of getting a value of T as large as the one we got or larger under the null hypothesis. Our test rejects when $p < \alpha$
  * **Power**: the probability that the null hypothesis is rejected when false, $1-\Beta$
  * Test statistic, statistic we are using as a test of the hypothesis. A function of sample we are going to test - eg. LRT
  * Null distribution: the probability distribution of the test statistic when the null hypothesis is true

1. Construct a test statistic T from our data, eg. for  $H_0: \mu = 0,\,H_1:\mu \neq 0$, take $T=|\bar{X}|$
2. Construct probability of exceeding critical value, set an alpha for the test.
3. Either compute critical value from PDF of x and alpha (don’t need x) or compute p-value from x and its PDF (don’t need alpha)
4. Determine if we reject null or not

* A simple hypothesis: $H_0$ and $H_1$ each have parameter equal to a specific value, specify a complete probability distribution. If a hypothesis does not completely specify the probability distribution, the hypothesis is called a composite hypothesis. This means a hypothesis like this is Poisson distributed or not is composite, because the null hypothesis needs to be a specific distribution with specified parameter to be simple. It is convention to choose the simpler hypothesis to be the null.

* Neyman Pearson Lemma: Suppose that $H_0$ and $H_A$ are simple hypotheses and that the test that rejects $H_0$  whenever the likelihood ratio is less than c and significance level $\alpha$. Then any other test for which the significance level is less than or equal to $\alpha$ has power less than or equal to that of the likelihood ratio test.

* For simple hypothesis: We write down the likelihood ratio and observe that small values of it correspond in a one-to-one manner with extreme values of a test statistic, in this case X. Knowing the null distribution of the test statistic makes it possible to choose a critical level that  produces a desired significance level $\alpha$.

* p-value is the smallest alpha for which we reject the null hypothesis 

  * critical value < test stat $\iff$ p-value < alpha
  * The p-value is the probability of a result as or more extreme than that actually observed if the null hypothesis were true.


  ![hypothesis_test_regions](/Users/spencerbraun/Documents/Notes/Stanford/STATS200/hypothesis_test_regions.png)

*  UMP: If the alternative $H_1$ is composite, a test that is most powerful for every simple alternative  in $H_1$ is said to be uniformly most powerful. 

   * In typical composite situations, there is no uniformly most powerful test. 

* The alternatives $H_1 : \mu < \mu_0$ and $H_1 : \mu> \mu_0$ are called one-sided alternatives. The  alternative $H_1 : \mu = \mu_0$ is a two-sided alternative. 

* Confidence Intervals: $\mu_0$ lies in the confidence interval for $\mu$ if and only if the hypothesis test accepts. In  other words, the confidence interval consists precisely of all those values of $\mu_0$ for  which the null hypothesis $H_0: \mu= \mu_0$ is accepted.

* Hypothesis testing with samples:

  * $\operatorname{Pr}\left(\bar{X}-\frac{s t_{(n-1)}^{1-\alpha / 2}}{\sqrt{n}} \leqslant \mu \leqslant \bar{X}+\frac{s t_{(n-1)}^{1-\alpha / 2}}{\sqrt{n}}\right)=1-\alpha$
  * Since symmetric t distribution: $\operatorname{Pr}\left(\bar{X}+\frac{s t_{(n-1)}^{\alpha / 2}}{\sqrt{n}} \leqslant \mu \leqslant \bar{X}+\frac{s t_{(n-1)}^{1-\alpha / 2}}{\sqrt{n}}\right)=1-\alpha$
  * 2 Sample t-test estimating $\mu_x - \mu_y$ by $\bar{X} - \bar{Y}$. 
    * $\bar{X} - \bar{Y} \sim N(\mu_x - \mu_y, \sigma^2(1/n + 1/m))$
    * $s^2_{pooled} = \frac{(n-1)s^2_x + (m-1)s^2_y}{n+m -2}$
    * $\hat{\Delta} \pm t^{1-\alpha / 2} s_{\text {pooled }} \sqrt{1 / n+1 / m}$

##### Bayes Testing / LRT

* For simple hypotheses. Likelihood Ratio - For two distributions, the probability of observing an event from one over the other.
* $LR = \frac{P(\text{all our data}|H_0)}{P(\text{all our data}|H_1)}$

1. Use density functions of data in the LR
2. Plug in the parameters under $H_0$ and $H_1$, eg $\mu_1, \mu_2$
3. Simplify LR and determine how LR changes as a function of X
4. Assume X is distributed by the density given in $H_0$. We use this density in the test.
5. Set LRT greater than some constant critical value and determine CI for given alpha using some known distribution approximation (eg. normal, chi-square)

* This example is typical of the way that the Neyman-Pearson Lemma is used. We write down the likelihood ratio and observe that small values of it correspond in a one-to-one manner with extreme values of a test statistic, fore example $\bar{X}$. Knowing  the null distribution of the test statistic makes it possible to choose a critical level that  produces a desired significance level $\alpha$. 
* The evidence provided by the data is contained in the likelihood ratio, which is multiplied by the ratio of prior probabilities to produce the ratio of posterior probabilities.

##### Generalized Likelihood Ratio Tests

* $\Lambda=\frac{\max _{\theta \in \omega_0 }[\operatorname{lik}(\theta)]}{\max _{\theta \in \Omega}[\operatorname{lik}(\theta)]} \leq \lambda_0$ - Testing the parameter space for observations against some cutoff value. $\Lambda \leq 1$ since numerator is a subset of the denominator
* We have to work out the distribution of $\Lambda$ under $H_0$ to get a significance level. This can be quite hard, so for iid data, we approximate using $-2log(\Lambda) \approx \chi^2_{(d)}$ under $H_0$ as $n \rightarrow \infty$. 
  * DoF d = # free parameters in H1 - # free parameters in H0. Example: N(2, 1) has 0 free parameters, $N(\sigma^2 + 1, \sigma^2)$ has 1, and $N(\mu, \sigma^2)$ has 2.
  * Intuition: the more free parameters you allow, the more the alternative can fit to the data and explain it. Therefore $-2log(\Lambda)$ must be larger to provide evidence against the null.
* For $H_0: \mu = \mu_0,\,H_1:\mu \neq \mu_0$, the numerator of the likelihood ratio is the density function at point $\mu_0$ and the denominator we plug in the mle of the parameter $\mu$, (so here for example, $\bar{X}$). This follows from the definition, since the maximum of the likelihood function over the parameter space is the MLE.
* Knowing the null distribution of the test statistic makes possible the construction of a rejection region for any significance level $\alpha$. Using the chi-square, RR is given by $\left|\bar{X}-\mu_{0}\right| \geq \frac{\sigma}{\sqrt{n}} z(\alpha / 2)$

* Under smoothness conditions on the probability density or frequency functions  involved, the null distribution of $−2 log \Lambda$ tends to a chi-square distribution  with degrees of freedom equal to  $dim\,\Omega−dim\,ω_0$ as the sample size tends to  infinity.
* Generally - set up the distributions, compose the likelihood ratio, often take the log of both sides, try to determine how the likelihood changes for different observed values.
* $-2 \log \Lambda \approx \sum_{i=1}^{m} \frac{\left[x_{i}-n p_{i}(\hat{\theta})\right]^{2}}{n p_{i}(\hat{\theta})} = X^2$ : RHS is Pearson’s test statistic for goodness of fit. Degrees of freedom (df): # of free parameters (think: # of unknown parameters) 

##### Pearson Chi-Square Test 

* Used on grouped data into bins
* $X^2 = \sum\frac{(Observed - Expected)^2}{Expected}$

* $X^{2}:=\sum_{1}^{n} \frac{\left(O_{i}-E_{i}\right)^{2}}{E_{i}} =\sum_{i=1}^{m} \frac{\left[x_{i}-n p_{i}(\hat{\theta})\right]^{2}}{n p_{i}(\hat{\theta})} \sim \chi_{\mathrm{d} f:=n-s-1}^{2}$
* $\begin{array}{l}{x^{2}=\text { Pearson's cumulative test statistic, which asymptotically approaches a } \chi^{2} \text { distribution. }} \\ {O_{i}=\text { the number of observations of type } i .} \\ {N=\text { total number of observations }} \\ {E_{i}=N p_{i}=\text { the expected (theoretical) count of type } i, \text { asserted by the null hypothesis that the fraction of type in the population is } p_{i}} \\ {n=\text { the number of cells in the table. }}\end{array}$
* Goodness-of-fit: whether eCDF differs from *any* theoretical CDF 

1. Build test statistic for n bins - $Oi$ is the observed count in each bin and $Ei$ is the expected count in each bin
2. Null hypothesis is O ~ E 
3. Compare the test statistic to critical values in the chi-square distribution
4. Degrees of freedom is number of cells less parameters being estimated.

##### Poisson Dispersion Test

* If one  has a specific alternative hypothesis in mind, better power can usually be obtained  by testing against that alternative rather than against a more general alternative.
* The two key assumptions underlying the Poisson distribution are that the rate  is constant and that the counts in one interval of time or space are independent of the counts in disjoint intervals. These conditions are often not met.
* Given counts $x_1,..., x_n$, we consider testing the null hypothesis that the counts  are Poisson with the common parameter $\lambda$ versus the alternative hypothesis that  they are Poisson but have different rates, $\lambda1,...,\lambda_n$.
* $\begin{aligned} \Lambda &=\frac{\prod_{i=1}^{n} \hat{\lambda}^{x_{i}} e^{-\hat{\lambda}} / x_{i} !}{\prod_{i=1}^{n} \tilde{\lambda}_{i}^{x_{i}} e^{-\bar{\lambda}_{i}} / x_{i} !} =\prod_{i=1}^{n}\left(\frac{\bar{x}}{x_{i}}\right)^{x_{i}} e^{x_{i}-\bar{x}} \end{aligned}$
* $-2 \log \Lambda \approx \frac{1}{\bar{x}} \sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}$ using Taylor approximation. 
* We use the above formula as the test statistic and find the relevant significance level / p-value from the poisson distribution.

##### Hanging Rootograms

* Hanging rootograms are a  graphical display of the differences between observed and fitted values in histograms. 
* Suppose we estimate $\bar{x} \rightarrow \mu, \, \hat{\sigma} \rightarrow \sigma$, the then probability that an observation falls in an interval between $x_{j-1}, x_j$ is $\hat{p}_{j}=\Phi\left(\frac{x_{j}-\bar{x}}{\hat{\sigma}}\right)-\Phi\left(\frac{x_{j-1}-\bar{x}}{\hat{\sigma}}\right)$. For sample size n, the predicted count in the jth interval is $\hat{n}_j = n\hat{p}_j$, which we then compare to the observed counts
* The hanging histogram is then the difference between the fitted frequency of the bins and the observed. Expect larger fluctuations in the center than in the tails since variance across buckets is not constant - can use a variance-stabilizing transformation. Often use $f(x) = \sqrt{x}$ so get a hanging rootogram showing $\sqrt{n_j} - \sqrt{\hat{n}_j}$
* From delta-method,  for $Y= f(X)$,  $\operatorname{Var}(Y) \approx \sigma^{2}(\mu)\left[f^{\prime}(\mu)\right]^{2}$. Variance is stabilized by making a transformation under which this is constant.
* Generally, deviations in the center  have been down-weighted and those in the tails emphasized by the transformation. 
* Hanging chi-gram: plots $\frac{n_j - \hat{n}_j}{\sqrt{\hat{n}_j}}$

##### Probability Plots

* Useful graphical tool for qualitatively assessing the  fit of data to a theoretical distribution
* Plotting the ordered observations against expected values ($E\left(X_{(j)}\right)=\frac{j}{n+1}$) (find order statistic of data and plot against quantiles of a known distribution)
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
  * Simply a reversal of the CDF for data consist of time until death or failure, chance of surviving past t.

###### Hazard Functions

* As the instantaneous death rate for individuals who  have survived up to a given time. $h(t)=\frac{f(t)}{1-F(t)} = \frac{f(t)}{S(t)} = -\frac{d}{dt}log(S(t))$
* May be thought of as the instantaneous rate of mortality for an individual alive at time t. If T is the lifetime of a manufactured component, it may be natural to think  of h(t) as the instantaneous or age-specific failure rate.

##### QQ Plots

* Shoes the ith order statistic $X_{(i)}$ against $F^{-1}(i/(n+1))$ for some distribution CDF F
* Pth quantile: $F(x) = p$ or $x_p = F^{-1}(p)$
* Plot the quantiles of one distribution against another
* Additive: if $y_{p}=x_{p}+h$, then for the quantiles $G(y)=F(y-h)$, for control group x with CDF F and treatment y with CDF G. For values of y G(y) = F(y) shifted to the right.
* Multiplicative: if $y_{p}=c x_{p}$, then for quantiles $G(y)=F(y / c)$
* If the values in the right tail would have to move down to the line that means they would have to become smaller and the data has a heavier tail than the distribution.
* To compare two batches of n numbers with order statistics X(1) ,..., X(n) and  Y(1) ,..., Y(n) , a Q-Q plot is simply constructed by plotting the points (X(i) , Y(i)). The difference from PP plots is simply that we are plotting two data quantiles now instead of data against a theoretical distribution.


##### Histograms, Density Curves, Stem / Leaf plots

* Let w(x) be a weight function: 
  * nonnegative 
  * symmetric
  * centered at zero
  * integrating to 1
* We rescale w: $w_{h}(x)=\frac{1}{h} w\left(\frac{x}{h}\right)$. Small h causes kernel function to be more peaked about 0, large h more spread out.
* Kernel probability density estimate: $f_{h}(x)=\frac{1}{n} \sum_{i=1}^{n} w_{h}\left(x-X_{i}\right)$ - superposition of hills centered on the observations. If $w_h(x)$ is standard normal $w_h(x - Xi)$ is normal with mean Xi and SD h. 
* The parameter h, bandwidth, controls the smoothness and is the bin width of the histogram. With histograms and density estimates, we lose information and cannot reconstruct the data.
*  Stem and leaf plots - retain numerical information while showing shape. 

##### Measures of Location

* Measure of the center of batch of numbers
* Arithmetic mean - sum over the count
* Robust measures  - insensitive to outliers, such as the median.
* When the data are a sample from a continuous probability law, the sample median can be viewed as an estimate of the population median. The distribution of the number of observations greater than the median is binomial  with n trials and probability 1/2 of success on each trial.
* Trimmed mean - The $100\alpha\%$  trimmed mean is easy to calculate: Order the data, discard the lowest $100\alpha\%$  and the highest $100\alpha\%$, and take the arithmetic mean of the remaining data: $\bar{x}_{\alpha}=\frac{x_{([n \alpha]+1)}+\cdots+x_{(n-[n \alpha])}}{n-2[n \alpha]}$
* M Estimates - minimizers of $\sum_{i=1}^{n} \Psi\left(\frac{X_{i}-\mu}{\sigma}\right)$. where the weight function is a compromise between the weight functions for the  mean and the median.
* Estimating variability of location estimates by bootstrap:
  * Suppose we denote the location estimate as $\hat{\theta}$; it is important to keep in  mind that $\hat{\theta}$ is a function of the random variables $X_1, X_2,..., X_n$ and hence has a probability distribution, its sampling distribution, which is determined by n and F. We don’t know F and $\hat{\theta}$ may be complicated.
  * We generate many samples from F (if we knew it) and calculate the value of $\hat{\theta}$, then could find measures on the samples like SD. Use empirical CDF instead as an approximation of F. 
  * A sample of size n from Fn is thus a sample of size n drawn with replacement from the collection $x_1, x_2,..., x_n$. We thus draw B samples of size n with replacement from the observed  data, producing $\theta_{1}^{*}, \theta_{2}^{*}, \ldots, \theta_{B}^{*}$.
  
  * Then the SD is estimated as: $s_{\hat{\theta}}=\sqrt{\frac{1}{B} \sum_{i=1}^{B}\left(\theta_{i}^{*}-\bar{\theta}^{*}\right)^{2}}$

##### Measures of Dispersion

* Sample standard deviation: $s^{2}=\frac{1}{n-1} \sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)^{2}$ (use n-1 as divisor to make s^2 unbiased estimate of population variance)
  * If sample from standard normal, $\frac{(n-1) s^{2}}{\sigma^{2}} \sim \chi_{n-1}^{2}$
* Median absolute deviation from the median (MAD): the data  are $x_1,..., x_n$ with median $\tilde{x}$, the MAD is defined to be the median of the numbers  $|x_i − \tilde{x}|$.
* These two measures of dispersion, the IQR and the MAD, can be converted into estimates of $\sigma$ for a normal distribution by dividing them by 1.35 and .675,  respectively.


##### Boxplot

* The range lines - A vertical line is drawn up from the upper quartile to the most extreme data point that is within a distance of 1.5 (IQR) of the upper quartile, ie $X_i > Q^{0.75} + 1.5\times IQR$  and $X_i < Q^{0.25} - 1.5\times IQR$. A similarly defined vertical line is drawn down from the lower quartile. Short horizontal lines are  added to mark the ends of these vertical lines. 

### Chapter 11 - Comparing Two Samples

#### Independent Samples

##### Normal Methods

* The observations from the control group are modeled as independent random  variables with a common distribution, F, and the observations from the treatment  group are modeled as being independent of each other and of the controls and as  having their own common distribution function, G.
* We will assume that a sample, $X_1,...,X_n$, is drawn from a normal distribution that has mean  $\mu_X$ and variance $\sigma^2$ , and that an independent sample,  $Y_1,...,Y_m$, is drawn from another normal distribution that has mean $\mu_Y$ and the same  variance, $\sigma^2$

* Estimate difference in means: $\bar{X}-\bar{Y} \sim N\left[\mu_{X}-\mu_{Y}, \sigma^{2}\left(\frac{1}{n}+\frac{1}{m}\right)\right]$, with CI for known variance: $(\bar{X}-\bar{Y}) \pm z(\alpha / 2) \sigma \sqrt{\frac{1}{n}+\frac{1}{m}}$
* Usually do not know variance, use pooled sample variance; $s_{p}^{2}=\frac{(n-1) s_{X}^{2}+(m-1) s_{Y}^{2}}{m+n-2}$
* Test statistic: $t=\frac{(\bar{X}-\bar{Y})-\left(\mu_{X}-\mu_{Y}\right)}{s_{p} \sqrt{\frac{1}{n}+\frac{1}{m}}} \sim t_{m+n-2}$. Standard error of $\bar{X} - \bar{Y}$: $s_{\bar{X}-\bar{Y}}=s_{p} \sqrt{\frac{1}{n}+\frac{1}{m}}$. Confidence interval: $(\bar{X}-\bar{Y}) \pm t_{m+n-2}(\alpha / 2) s_{\bar{X}-\bar{Y}}$
* Hypothesis testing: $H_{0}: \mu_{X}=\mu_{Y}, \,H_{1}: \mu_{X} \neq \mu_{Y}$. Uses test stat: $t=\frac{\bar{X}-\bar{Y}}{s_{\bar{X}-\bar{Y}}}$ since essentially testing if the difference has zero mean. If we wanted to test against a non-zero difference, might look something like $\mu_X - (\mu_Y + 5) = 0$ This test can be derived from the GLRT - rejects for large values.
* Without assumption of equal variances, estimate of $\operatorname{Var}(\bar{X}-\bar{Y}) = \frac{s_{X}^{2}}{n}+\frac{s_{Y}^{2}}{m}$  and we use the t-distribution with $\mathrm{df}=\frac{\left[\left(s_{X}^{2} / n\right)+\left(s_{Y}^{2} / m\right)\right]^{2}}{\frac{\left(s_{X}^{2} / n\right)^{2}}{n-1}+\frac{\left(s_{Y}^{2} / m\right)^{2}}{m-1}}$ 
* It is sometimes advocated that skewed data be transformed to a more symmetric  shape before normal theory is applied. Transformations such as taking the log or the square root can be effective in symmetrizing skewed distributions because they  spread out small values and compress large ones.
* The ratio of the  standard deviation of a distribution to the mean is called the coefficient of variation (CV); it expresses the standard deviation as a fraction of the mean.
* The power of the two-sample t test  depends on four factors: 
  1. The real difference $\triangle = |\mu_X −\mu_Y |$. The larger this difference, the greater the power. 
  2. The significance level $\alpha$ at which the test is done. Large alpha larger power
  3. The smaller the population standard deviation, the larger the power. 
  4. The sample sizes n and m. The larger the sample sizes, the greater the power. 

##### Nonparametric Method - Mann-Whitney

* Suppose that we have  m + n experimental units to assign to a treatment group and a control group. The  assignment is made at random: n units are randomly chosen and assigned to  the control, and the remaining m units are assigned to the treatment.
* First, we group all m + n  observations together and rank them in order of increasing size. We next calculate the sum of the ranks of those observations that came from the control group. If this sum is too small or too large, we will reject the null  hypothesis. We consult a rank table to determine the level of significance for the rank sum obtained for the group with the smaller rank sum. We have not made any assumption that the observations  from the control and treatment groups are samples from a probability distribution. 
* When it is more appropriate to model the control values, $X_1,...,X_n,$ as a sample  from some probability distribution F and the experimental values, $Y_1,..., Y_m$, as a  sample from some distribution G, the Mann-Whitney test is a test of the null hypothesis  $H_0: F = G$. The reasoning is exactly the same: Under $H_0$, any assignment of ranks  to the pooled m + n observations is equally likely.
* If the groups are roughly the same, then the ranks will have a good amount of alternation and be mixed from each group. The more separation in the ranks, the more different the two groups.
* Let $T_Y$ denote the sum of the ranks of $Y_1, Y_2,..., Y_m$.  $E(T_Y )$ and $Var(T_Y )$ under the null hypothesis F = G: $E\left(T_{Y}\right)=\frac{m(m+n+1)}{2}$, $\operatorname{Var}\left(T_{Y}\right)=\frac{m n(m+n+1)}{12}$
  * Under $H_0: F = G$, $E\left(U_{Y}\right)=\frac{m n}{2}$, $\operatorname{Var}\left(U_{Y}\right)=\frac{m n(m+n+1)}{12}$, and $\frac{U_{Y}-E\left(U_{Y}\right)}{\sqrt{\operatorname{Var}\left(U_{Y}\right)}} \sim N(0,1)$ for m, n over 10.
* Since the actual numerical values are replaced by their ranks, the test is insensitive to outliers, whereas the t test is sensitive. It has been shown that even when the assumption of normality holds, the Mann-Whitney test is nearly as powerful as the t test and it is thus generally preferable, especially for small sample  sizes. 
* Bootstrap: As before, suppose that $X_1, X_2,..., X_n$ and $Y_1, Y_2,..., Y_m$ are  two independent samples from distributions F and G, respectively, and that $pi=  P(X < Y )$ is estimated by $\hat{\pi}$. How can the standard error of $\hat{\pi}$ be estimated and how  can an approximate confidence interval for $\pi$ be constructed?
  * An approximation can be obtained by using the empirical distributions $F_n$ and $G_n$ in their places. This means that a bootstrap value of  $\hat{\pi}$ is generated by randomly selecting n values from $X_1, X_2,..., X_n$ with replacement,  m values from $Y_1, Y_2,..., Y_m$ with replacement and calculating the resulting value  of $\hat{\pi}$

##### Bayesian Approach

* The $X_i$ are i.i.d. normal with mean $\mu_X$ and precision $\xi$; and the $Y_j$ are i.i.d. normal with mean $\mu_Y$, precision $\xi$, and independent of the $X_i$
* We use improprer priors to get an approx result
* $\frac{\Delta-(\bar{x}-\bar{y})}{s_{p} \sqrt{n^{-1}+m^{-1}}} \sim t_{n+m-2}$ but here the mean difference and $s_p$ are fixed and $\triangle$ is random. The posterior can then be found using the t distribution: $=P\left(T \geq \frac{\bar{y}-\bar{x}}{s_{p} \sqrt{n^{-1}+m^{-1}}}\right)$

#### Comparing Paired Samples

* May match subjects with related characteristics, then assign one to the control and one to the experimental group.
* A pair is $(X_i, Y_i)$ and these are not assumed to be independent (think left and right hand strength), but for $i \neq j$, $(X_i, X_j)$ can be indepedent, think left hand strength across a population.
  * sampling two different people: $X_{i}-Y_{j} \sim N\left(\mu_{x}-\mu_{u}, \sigma_{x}^{2}+\sigma_{y}^{2}\right)$
  * sampling one person: $X_{i}-Y_{i} \sim N\left(\mu_{x}-\mu_{u}, \sigma_{x}^{2}+\sigma_{y}^{2}-2 \rho \sigma_{x} \sigma_{y}\right)$
* Pairs are $(X_i, Y_i), \, i=1,...,n$, different means and variances. Different pairs are independently distributed and $\operatorname{Cov}\left(X_{i}, Y_{i}\right)=\sigma_{X Y}$
* For $D_{i}=X_{i}-Y_{i}$, independent with $E\left(D_{i}\right)=\mu_{X}-\mu_{Y}$, $\begin{aligned} \operatorname{Var}\left(D_{i}\right) &=\sigma_{X}^{2}+\sigma_{Y}^{2}-2 \sigma_{X Y} =\sigma_{X}^{2}+\sigma_{Y}^{2}-2 \rho \sigma_{X} \sigma_{Y} \end{aligned}$

##### Normal Methods

* $\begin{aligned} E\left(D_{i}\right) &=\mu_{X}-\mu_{Y}=\mu_{D} , \, \operatorname{Var}\left(D_{i}\right) =\sigma_{D}^{2} \end{aligned}$
* $D_i \sim N\left(\mu_{x}-\mu_{y}, \sigma_{x}^{2}+\sigma_{y}^{2}-2 \rho \sigma_{x} \sigma_{y}\right) = N\left(\mu_{D}, \sigma_{D}^{2}\right)$
* Test stat: $t=\frac{\bar{D}-\mu_{D}}{s_{\bar{D}}}$, CI: $\bar{D} \pm t_{n-1}(\alpha / 2) s_{\bar{D}}$, Two sided RR: $|\bar{D}|>t_{n-1}(\alpha / 2) s_{\bar{D}}$

##### Nonparametric Method - Signed Rank Test

* Test statistic:
  * Calculate differences $D_i$ and rank by absolute values of the diffs
  * Restore the signs of the differences to the ranks
  * Calculate $W_+$, the sum of ranks that have positive signs
* If one condition produces larger values than the other, $W_+$ will take on extreme values. We test the null hypothesis that the distribution of $D_i$ is symmetric about zero.

### Chapter 12 - ANOVA

#### One Way Layout

* An experimental design in which independent measurements  are made under each of several treatments.

##### Normal Theory / F-Test

* Wefirst discuss the analysis of variance and the F test in the case of I groups, each  containing J samples. The I groups will be referred to generically as treatments, or levels
* Let $Y_{ij} = $ the jth observation of the ith treatment
* $Y_{i j}=\mu+\alpha_{i}+\varepsilon_{i j}$ 
  * observations are corrupted by random independent errors $\epsilon_{ij}$, normally distributed with mean zero and constant variance $\sigma^2$
  * F test is approximately valid for large non-normal samples
  * $\mu$ is the overall mean level
  * $\alpha_i$ is the differential effect of the ith treatment normalized st $\sum_{i=1}^{l} \alpha_{i}=0$.
* Expected response to ith treatment: $E\left(Y_{i j}\right)=\mu+\alpha_{i}$ 
* The total sum of squares equals the sum of squares within groups plus the sum of square between groups: $S S_{T O T}=S S_{W}+S S_{B}$
* Lemma A: For $X_i$ $\perp$ RVs with means $\mu_i$ and shared variance $\sigma^2$ then $E\left(X_{i}-\bar{X}\right)^{2}=\left(\mu_{i}-\bar{\mu}\right)^{2}+\frac{n-1}{n} \sigma^{2}$
* Then Theorem A: $E\left(S S_{W}\right)=\sum_{i=1}^{I} \sum_{j=1}^{J} E\left(Y_{i j}-\bar{Y}_{i .}\right)^{2} =\sum_{i=1}^{I} \sum_{j=1}^{J} \frac{J-1}{J} \sigma^{2}=I(J-1) \sigma^{2}$. Can use for unbiased estimate of $\sigma^2: s_{p}^{2}=\frac{S S_{w}}{I(J-1)}$
*  Theorem B: For independent, N(0, $\sigma^2$) errors, $S S_{W} / \sigma^{2} \sim \chi^2_{I(J-1)}$. If $\alpha_i = 0 \, \forall i$, $S S_{B} / \sigma^{2} \sim \chi^2_{I-1}$ and $SS_W \perp SS_B$
* We use test statistic $F=\frac{S S_{B} /(I-1)}{S S_{W} /[I(J-1)]}$ to test $H_{0}: \alpha_{1}=\alpha_{2}=\cdots=\alpha_{l}=0$. The denominator of the F statistic has expected value equal to $\sigma^2$ ,  and the expectation of the numerator is $J (I−1)^{−1} \sum_{i=1}^I \alpha_i^2 + \sigma^2 2$ . Thus, if the null hypothesis is true, the F statistic should be close to 1, whereas if it is false, the statistic should be larger. Reject hypothesis for large values of F. 
* Under normally distributed errors, null distribution of $F \sim F_{(I-1),(I(J-1))}$
* For unequal number of observations under various treatments:
  * $E\left(S S_{w}\right)=\sigma^{2} \sum_{i=1}^{I}\left(J_{i}-1\right)$, $E\left(S S_{B}\right)=(I-1) \sigma^{2}+\sum_{i=1}^{I} J_{i} \alpha_{i}^{2}$

##### Problem of Multiple Comparisons

* Real interest may be focused on  comparing pairs or groups of treatments and estimating the treatment means and  their differences - the F-test does not tell us how our treatment effects differ.
* Tukey’s Method