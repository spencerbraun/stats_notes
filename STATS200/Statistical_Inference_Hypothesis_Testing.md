---
title: Statistical Inference - Hypothesis Testing
author: Spencer Braun
date: 20191016
---

[TOC]

<style type="text/css">@page { size: letter; margin: 0.25in; font-size: 20px; }</style>

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

* Two sample shift model: $X_{1}, X_{2}, \ldots, X_{n_{1}}$ is a random sample from distribution F(X) and $Y_{1}, Y_{2}, \ldots, Y_{n_{2}}$ is a random sample from $G(y)=F(y-\theta)$ for an unknown $\theta$. The null is that the distributions are equal, ie $\theta = 0$

##### Normal Methods

* The observations from the control group are modeled as independent random  variables with a common distribution, F, and the observations from the treatment  group are modeled as being independent of each other and of the controls and as  having their own common distribution function, G.
* We will assume that a sample, $X_1,...,X_n$, is drawn from a normal distribution that has mean  $\mu_X$ and variance $\sigma^2$ , and that an independent sample,  $Y_1,...,Y_m$, is drawn from another normal distribution that has mean $\mu_Y$ and the same  variance, $\sigma^2$
* Estimate difference in means: $\bar{X}-\bar{Y} \sim N\left[\mu_{X}-\mu_{Y}, \sigma^{2}\left(\frac{1}{n}+\frac{1}{m}\right)\right]$, with CI for known variance: $(\bar{X}-\bar{Y}) \pm z(\alpha / 2) \sigma \sqrt{\frac{1}{n}+\frac{1}{m}}$
* Usually do not know variance, use pooled sample variance; $s_{p}^{2}=\frac{(n-1) s_{X}^{2}+(m-1) s_{Y}^{2}}{m+n-2}$
* Test statistic: $t=\frac{(\bar{X}-\bar{Y})-\left(\mu_{X}-\mu_{Y}\right)}{s_{p} \sqrt{\frac{1}{n}+\frac{1}{m}}} \sim t_{m+n-2}$. Standard error of $\bar{X} - \bar{Y}$: $s_{\bar{X}-\bar{Y}}=s_{p} \sqrt{\frac{1}{n}+\frac{1}{m}}$. Confidence interval: $(\bar{X}-\bar{Y}) \pm t_{m+n-2}(\alpha / 2) s_{\bar{X}-\bar{Y}}$
* Hypothesis testing: $H_{0}: \mu_{X}=\mu_{Y}, \,H_{1}: \mu_{X} \neq \mu_{Y}$. Uses test stat: $t=\frac{\bar{X}-\bar{Y}}{s_{\bar{X}-\bar{Y}}}$ since essentially testing if the difference has zero mean. If we wanted to test against a non-zero difference, might look something like $\mu_X - (\mu_Y + 5) = 0$ This test can be derived from the GLRT - rejects for large values.
* Without assumption of equal variances, estimate of $\operatorname{Var}(\bar{X}-\bar{Y}) = \frac{s_{X}^{2}}{n}+\frac{s_{Y}^{2}}{m}$  and we use the t-distribution with $\mathrm{df}=\frac{\left[\left(s_{X}^{2} / n\right)+\left(s_{Y}^{2} / m\right)\right]^{2}}{\frac{\left(s_{X}^{2} / n\right)^{2}}{n-1}+\frac{\left(s_{Y}^{2} / m\right)^{2}}{m-1}}$ 
* Procedure:
  * Determine distribution of $\bar{X}-\bar{Y}$ and build normalized test statistic Z. 
  * If $\sigma$ is unknown, calculate $s_p$ and build test statistic t ($t=\frac{\bar{X}-\bar{Y}}{s \bar{x}-\bar{Y}}$ for $H_0 = 0$). If variances are unequal, $\operatorname{Var}(\bar{X}-\bar{Y}) = \frac{s_{X}^{2}}{n}+\frac{s_{Y}^{2}}{m}$ and modify t distribution degrees of freedom
  * Hypothesis test against null or $(\bar{X}-\bar{Y}) \pm t_{m+n-2}(\alpha / 2) s_{\bar{X}-\bar{Y}}$
  * Power: For alternative hypothesis $H_{1}: \mu_{X}-\mu_{Y}=\Delta$, calculate power using the normal distribution (since special noncentral t tables would be need otherwise). Then RHS power is given by $1-\Phi\left[z(\alpha / 2)-\frac{\Delta}{\sigma} \sqrt{\frac{n}{2}}\right]$ ie. the (z stat used for the RR) -  (mean difference)/(pooled SE). Total two tailed power: $1-\Phi\left[z(\alpha / 2)-\frac{\Delta}{\sigma} \sqrt{\frac{n}{2}}\right]+\Phi\left[-z(\alpha / 2)-\frac{\Delta}{\sigma} \sqrt{\frac{n}{2}}\right]$
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
* When it is more appropriate to model the control values, $X_1,...,X_n,$ as a sample from some probability distribution F and the experimental values, $Y_1,..., Y_m$, as a  sample from some distribution G, the Mann-Whitney test is a test of the null hypothesis  $H_0: F = G$. The reasoning is exactly the same: Under $H_0$, any assignment of ranks  to the pooled m + n observations is equally likely.
* If the groups are roughly the same, then the ranks will have a good amount of alternation and be mixed from each group. The more separation in the ranks, the more different the two groups.
* $U=n m+\frac{n\left(n+1\right)}{2}-W$, for W = rank sum for sample X. U is obtained by ordering all n+m observations and counting the number of observations in sample I that precede each observation in sample II - U is the sum of these counts
* Let $T_Y$ denote the sum of the ranks of $Y_1, Y_2,..., Y_m$.  $E(T_Y )$ and $Var(T_Y )$ under the null hypothesis F = G: $E\left(T_{Y}\right)=\frac{m(m+n+1)}{2}$, $\operatorname{Var}\left(T_{Y}\right)=\frac{m n(m+n+1)}{12}$
  * Under $H_0: F = G$, $E\left(U_{Y}\right)=\frac{m n}{2}$, $\operatorname{Var}\left(U_{Y}\right)=\frac{m n(m+n+1)}{12}$, and $\frac{U_{Y}-E\left(U_{Y}\right)}{\sqrt{\operatorname{Var}\left(U_{Y}\right)}} \sim N(0,1)$ for m, n over 10.
* Since the actual numerical values are replaced by their ranks, the test is insensitive to outliers, whereas the t test is sensitive. It has been shown that even when the assumption of normality holds, the Mann-Whitney test is nearly as powerful as the t test and it is thus generally preferable, especially for small sample  sizes. 
* Bootstrap: As before, suppose that $X_1, X_2,..., X_n$ and $Y_1, Y_2,..., Y_m$ are  two independent samples from distributions F and G, respectively, and that $pi=  P(X < Y )$ is estimated by $\hat{\pi}$. How can the standard error of $\hat{\pi}$ be estimated and how  can an approximate confidence interval for $\pi$ be constructed?
  * An approximation can be obtained by using the empirical distributions $F_n$ and $G_n$ in their places. This means that a bootstrap value of  $\hat{\pi}$ is generated by randomly selecting n values from $X_1, X_2,..., X_n$ with replacement,  m values from $Y_1, Y_2,..., Y_m$ with replacement and calculating the resulting value  of $\hat{\pi}$
* Procedure
  * Group all m + n  observations together and rank them in order of increasing size
  * Calculate the sum of the ranks of those observations. Take the smaller rank sum
  * Under the null, every assignment of m+n ranks to observations is equally likely, hence each of the ${m+n \choose m }$ assignments to the control group is equally likely. 
  * Look up the smaller rank sum in a table to determine if we reject.
  * When we assume a distribution for control variables $X \ sim F$ and experimental variables $Y \sim G$, M-W is a test of null $H_0: F = G$
  * We can for larger samples normalize the test stat sum of ranks T $\frac{T-E(T)}{\sigma_{T}}$ to use normal distribution
* CI’s in a Shift Model $G(x)=F(x-\Delta)$
  * Confidence interval for $\Delta$
  * 

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
* Procedure
  * Calculate differences in paired samples - say before and after a treatment for each patient: $\bar{D}$
  * Build test statistic $t=\frac{\bar{D}-\mu_{D}}{s_{\bar{D}}}$, use $t_{n-1}$ to find critical value

##### Nonparametric Method - Sign Test

* n pairs of observations $(X_i, Y_i)$, null is that the distributions of X and Y are the same. The probability that the mean difference $D_i > 0 = 0.5$ If M is the total number of positive differences, under the null $M \sim Bin(n, 1/2)$

1. Let $p = P(X>Y)$
2. $\begin{array}{l}{\text { Null hypothesis: } H_{0}: p=1 / 2}  {\text { Alternative hypothesis: } H_{a}: p>1 / 2 \text { or }(p<1 / 2 \text { or } p \neq 1 / 2)}\end{array}$
3. Test statistic: M = # of positive $D_i = X_i - Y_i$
4. Rejection region for $H_a: p > 1/2$, then reject for largest values of M
5. Assumptions: pairs are randomly and independently selected

* When we encounter ties due to equal observations in pairs, can delete and reduce n. If n is large, can use normal approximation $Z = \frac{M - np}{\sqrt{npq}}$

##### Nonparametric Method - Wilcoxon Signed Rank Test

* Under the null of equal distributions, if we were to order the differences according to their absolute values and rank them from smallest to largest, the expected rank sums for the negative and positive differences would be equal.
* Procedure
  * Calculate differences $D_i$, removing differences equal to 0. Under the null D is symmetric about 0
  * Rank by absolute values of the diffs, averaging the ranks for tied differences
  * Restore the signs of the differences to the ranks
  * Calculate $W_+$, the sum of ranks that have positive signs. Use the smaller sum as our test statistic - the smaller the value of the statistic, the greater the weight of evidence favoring rejection. Use W- to detect shifts of Y to the left of X; W+ to detect shifts of Y to the right of X
  * For two tailed test $T=\min \left(T^{+}, T^{-}\right)$. Reject null $H_{0} \text { if } T \leq T_{0}$ for CV of two tailed test.
  * For larger samples, a normal approximation of the null distribution can be used, using the expectation and variance below.
* If one condition produces larger values than the other, $W_+$ will take on extreme values. We test the null hypothesis that the distribution of $D_i$ is symmetric about zero. Equivalent then to Bern(0.5). $E(W_+) = \frac{n(n+1)}{4},\, Var(W_+) = \frac{n(n+1)(2n+1)}{24}$. For large samples $z=\frac{T^{+}-E\left(T^{+}\right)}{\sqrt{V\left(T^{+}\right)}}=\frac{T^{+}-[n(n+1) / 4]}{\sqrt{n(n+1)(2 n+1) / 24}}$

### Chapter 12 - ANOVA

#### One Way Layout

* An experimental design in which independent measurements  are made under each of several treatments. Analogous to the two independent samples tests in chapter 11.

##### Normal Theory / F-Test

* We first discuss the analysis of variance and the F test in the case of I groups, each containing J samples. The I groups will be referred to generically as treatments, or levels
* Let $Y_{ij} = $ the jth observation of the ith treatment
* $Y_{i j}=\mu+\alpha_{i}+\varepsilon_{i j}$ 
  * observations are corrupted by random <u>independent</u> errors $\epsilon_{ij}$, <u>normally distributed</u> with mean zero and <u>constant variance</u> $\sigma^2$
  * F test is approximately valid for large non-normal samples
  * $\mu$ is the overall mean level
  * $\alpha_i$ is the differential effect of the ith treatment normalized st $\sum_{i=1}^{l} \alpha_{i}=0$.
* Expected response to ith treatment: $E\left(Y_{i j}\right)=\mu+\alpha_{i}$ 
* The total sum of squares equals the sum of squares within groups plus the sum of square between groups: $S S_{T O T}=S S_{W}+S S_{B} \rightarrow \sum_{i=1}^{I} \sum_{j=1}^{J}\left(Y_{i j}-\bar{Y}_{. .}\right)^{2}=\sum_{i=1}^{I} \sum_{j=1}^{J}\left(Y_{i j}-\bar{Y}_{i .}\right)^{2}+J \sum_{i=1}^{I}\left(\bar{Y}_{i .}-\bar{Y}_{. .}\right)^{2}$
* Lemma A: For $X_i$ $\perp$ RVs with means $\mu_i$ and shared variance $\sigma^2$ then $E\left(X_{i}-\bar{X}\right)^{2}=\left(\mu_{i}-\bar{\mu}\right)^{2}+\frac{n-1}{n} \sigma^{2}$
* Then Theorem A: $E\left(S S_{W}\right)=\sum_{i=1}^{I} \sum_{j=1}^{J} E\left(Y_{i j}-\bar{Y}_{i .}\right)^{2} =I(J-1) \sigma^{2}$. 
  * Can use for unbiased estimate of $\sigma^2: s_{p}^{2}=\frac{S S_{w}}{I(J-1)}$, $S S_{W}=\sum_{i=1}^{I}(J-1) s_{i}^{2}$
  * If all $\alpha_i$ are zero, then expected SSw and SSb should be about equal, if some alphas are not zero then SSb increases.
*  Theorem B: For independent, N(0, $\sigma^2$) errors, $S S_{W} / \sigma^{2} \sim \chi^2_{I(J-1)}$. If $\alpha_i = 0 \, \forall i$, $S S_{B} / \sigma^{2} \sim \chi^2_{I-1}$ and $SS_W \perp SS_B$
* Procedure
  * We use test statistic $F=\frac{S S_{B} /(I-1)}{S S_{W} /[I(J-1)]}$ to test $H_{0}: \alpha_{1}=\alpha_{2}=\cdots=\alpha_{l}=0$. The denominator of the F statistic has expected value equal to $\sigma^2$ ,  and the expectation of the numerator is $J (I−1)^{−1} \sum_{i=1}^I \alpha_i^2 + \sigma^2 2$ . Thus, if the null hypothesis is true, the F statistic should be close to 1, whereas if it is false, the statistic should be larger. Reject hypothesis for large values of F. 
  * Under normally distributed errors, null distribution of $F \sim F_{(I-1),(I(J-1))}$
* For unequal number of observations under various treatments:
  * $E\left(S S_{w}\right)=\sigma^{2} \sum_{i=1}^{I}\left(J_{i}-1\right)$, $E\left(S S_{B}\right)=(I-1) \sigma^{2}+\sum_{i=1}^{I} J_{i} \alpha_{i}^{2}$

##### Kruskal-Wallis Test

* We assume that independent random samples have been drawn from k populations that differ only in location. The samples sizes may be unequal, and $n_i$ is the sample size drawn from the ith population. Combine all samples into n observations and rank them from smallest to largest. Tied ranks are averaged.
* $R_i$ = sum of ranks for observations from population i, $\bar{R}_{i}=R_{i} / n_{i}$ is the average of the ranks. If the null hypothesis is true and the populations do not differ in location, we would expect the $\bar{R}_i$ values to be approximately equal and $V=\sum_{i=1}^{k} n_{i}\left(\bar{R}_{i}-\bar{R}\right)^{2} =\sum_{i=1}^{k} n_{i}\left(\bar{R}_{i}-\frac{n+1}{2}\right)^{2}$ should be small. 
* K-W use test statistic $H=\frac{12}{n(n+1)} \sum_{i=1}^{k} \frac{R_{i}^{2}}{n_{i}}-3(n+1)$. Then $P[H>h(\alpha)]=\alpha$. For large enough $n_i$ (say 5), $H \sim \chi^2_{k-1}$. Then we reject $H_0$ if  $H>\chi_{\alpha}^{2} \text { with }(k-1) \text { df. }$ 
* Using Rice notation:
  * $R_{i j}=$ = the rank of $Y_{ij}$ in the combined sample 
  * $\bar{R}_{i .}=\frac{1}{J_{i}} \sum_{j=1}^{J_{i}} R_{i j}$, $\bar{R}_{..} = \frac{N+1}{2}$
  * $S S_{B}=\sum_{i=1}^{I} J_{i}\left(\bar{R}_{i .}-\bar{R}_{. .}\right)^{2}$
  * $K=\frac{12}{N(N+1)}\left(\sum_{i=1}^{I} J_{i} \bar{R}_{i .}^{2}\right)-3(N+1)$

##### Problem of Multiple Comparisons

* Real interest may be focused on  comparing pairs or groups of treatments and estimating the treatment means and  their differences - the F-test does not tell us how our treatment effects differ.
* Tukey’s Method - to construct confidence intervals for the differences of all pairs of means in such a way that the intervals simultaneously have a set coverage probability, then rely on the duality of tests and CIs
  * $\max _{i_{1}, i_{2}} \frac{\left|\left(\bar{Y}_{i_{1} .}-\mu_{i_{1}}\right)-\left(\bar{Y}_{i_{2} .}-\mu_{i_{2}}\right)\right|}{s_{p} / \sqrt{J}}$ random variable where max is taken over all pairs. Called the studentized range distribution with parameters I (number of samples being compared) and I(J-1) (df of $s_p$). 
  * CI constructed as $\left(\bar{Y}_{i_{1} .}-\bar{Y}_{i_{2} .}\right) \pm q_{I, I(J-1)}(\alpha) \frac{s_{p}}{\sqrt{J}}$, reject if $\left|\bar{Y}_{i_{1} .}-\bar{Y}_{i_{2} .}\right|>q_{I, I(J-1)}(\alpha) \frac{s_{p}}{\sqrt{J}}$
  * Essentially calculate $q_{I, I(J-1)}(\alpha) \frac{s_{p}}{\sqrt{J}}$ and see if there are differences in means of different labs that exceed this value 
* Bonferroni Method: Desired error rate $\alpha$ can be obtained over k null hypotheses by testing each null hypothesis at level $\alpha/k$. Advantage of not needing the sample sizes for each treatment

#### Two Way Layout

* A two-way layout is an experimental design involving two factors, each at two or more levels. The levels of one factor might be various drugs, for example, and the levels of the other factor might be genders. If there are I levels of one factor and  J of the other, there are I × J combinations. Assume K independent observations are taken for each of these combinations
* Think of 3 electric ranges cooking over 3 menus - 9 total means to compare

##### Additive Parametrization

* $\hat{Y}_{i j}=\hat{\mu}+\hat{\alpha}_{i}+\hat{\beta}_{j}$ 
* Calculate total average. Calculate the means for each I group across J. Differential effects $\hat{\alpha_i}$ are the differences between the group means and overall mean. Calculate the means for each group J across I, and repeat the calculation of differential effects $\hat{\beta}_j$. 
* The differences of the observed values and the fitted values, $Y_{i j}-\hat{Y}_{i j}$ are the residuals from the additive model. $\sum_{i=1}^{3} \hat{\delta}_{i j}=\sum_{j=1}^{3} \hat{\delta}_{i j}=0$ for residuals $\delta_{ij} = Y_{i j}-\hat{\mu}-\hat{\alpha}_{i}-\hat{\beta}_{j}$. Then the model $Y_{i j}=\hat{\mu}+\hat{\alpha}_{i}+\hat{\beta}_{j}+\hat{\delta}_{i j}$ fits the data points exactly.

##### Normal Theory for Two-Way

* Balanced design - equal number of observations per cell, and we assume K > 1 per cell
* $Y_{i j k}=\mu+\alpha_{i}+\beta_{j}+\delta_{i j}+\varepsilon_{i j k}$ for $Y_{ijk}$, the kth observation in cell ij. $\epsilon_{ijk}$ are random errors iid $N(0, \sigma^2)$ (common variance). 
* Therefore $E\left(Y_{i j k}\right)=\mu+\alpha_{i}+\beta_{j}+\delta_{i j}$ and the parameters are constrained by $\begin{array}{l}{\sum_{i=1}^{I} \alpha_{i}=0} ,\, {\sum_{j=1}^{J} \beta_{j}=0} ,\, {\sum_{i=1}^{I} \delta_{i j}=\sum_{j=1}^{J} \delta_{i j}=0}\end{array}$
* Use MLE on the unknown parameters. Comparing sum of squares we get $S S_{T O T}=S S_{A}+S S_{B}+S S_{A B}+S S_{E}$
  * Under error assumptions above:
  * $\begin{array}{l}{E\left(S S_{A}\right)=(I-1) \sigma^{2}+J K \sum_{i=1}^{I} \alpha_{i}^{2}} \\ {E\left(S S_{B}\right)=(J-1) \sigma^{2}+I K \sum_{j=1}^{J} \beta_{j}^{2}} \\ {E\left(S S_{A B}\right)=(I-1)(J-1) \sigma^{2}+K \sum_{i=1}^{I} \sum_{j=1}^{J} \delta_{i j}^{2}} \\ {E\left(S S_{E}\right)=I J(K-1) \sigma^{2}}\end{array}$
* $S S_{E} / \sigma^{2} \sim \chi^2_{IJ(K-1)}$
* Under null $H_{A}: \alpha_{i}=0, i=1, \ldots, I$, $S S_{A} / \sigma^{2} \sim \chi^2_{I-1}$
* Under null $H_{B}: \beta_{j}=0, \quad j=1, \ldots, J$, $S S_{B} / \sigma^{2} \sim \chi^2_{J-1}$
* Under null $H_{A B}: \delta_{i j}=0, i=1, \ldots, I, j=1, \ldots, J$, $S S_{A B} / \sigma^{2} \sim \chi^2_{(I-1)(J-1)}$
* F tests of these null hypotheses are conducted by comparing the appropriate SS to the sum of squares for error.

### Chapter 13 - Analysis of Categorical Data

* Two way tables - suppose rows are hair colors and columns eye colors, each cell is a count of people who fall in that cross classification - can we find a relationship between hair color and eye color?

##### Fisher’s Exact Test

* According to the null hypothesis, the margins of the table are fixes - ie the overall counts for each category. The randomization determines the counts in the interior of the table (capital letters) subject to the margin constraints, leaving us with one degree of freedom. 
* Under the null $N_{11}$ is distributed as the number of successes in 24 draws without replacement from a population of 35 successes and 13 failures - hypergeometric. The probability $N_{11} = n_{11} = p\left(n_{11}\right)=\frac{\left(\begin{array}{l}{n_{1 .}} \\ {n_{11}}\end{array}\right)\left(\begin{array}{l}{n_{2 .}} \\ {n_{21}}\end{array}\right)}{\left(\begin{array}{l}{n_{. .}} \\ {n .1}\end{array}\right)}$
* We use $N_{11}$ as the test statistic to test the null.

##### Chi-Square Test of Homogeneity

* Suppose that we have independent observations from J multinomial distributions,  each of which has I cells, and that we want to test whether the cell probabilities  of the multinomials are equal—that is, to test the homogeneity of the multinomial  distributions. 
* Example: how close is an admirer to matching Jane Austen’s style using word counts in their works
* The six word counts for Sense and Sensibility will  be modeled as a realization of a multinomial random variable with unknown cell  probabilities and total count 375; the counts for the other works will be similarly  modeled as independent multinomial random variables. 
* Thus, we must consider comparing J multinomial distributions each having I categories. If the probability of the ith category of the jth multinomial is denoted $\pi_{ij}$ ,  the null hypothesis to be tested is: $H_{0}: \pi_{i 1}=\pi_{i 2}=\cdots=\pi_{i J}, \quad i=1, \ldots, I$. This is essentially a goodness of fit test
* Under H0, each of the J multinomials has the same probability for the ith category, say $\pi_i$ . The following theorem shows that the mle of $\pi_i$ is simply $ni. /n..$. Here, $n_{i.}$ is the total count in the ith category, $n_{..}$ is the grand  total count, $n_{.j}$ is the total count for the jth multinomial. 
* $E_{i j}=\frac{n_{. j} n_{i .}}{n ..}$ for the jth multinomial the expected count in the ith category is the estimated probability of that cell times the total number of observation for the jth multinomial. Can use Pearson Chi-Square: $X^2 =\sum_{i=1}^{I} \sum_{j=1}^{J} \frac{\left(n_{i j}-n_{i .} n_{. j} / n_{. .}\right)^{2}}{n_{i . }n_{ . j} / n_{. .}}$ with $d f=(I-1)(J-1)$

##### Chi-Square Test of Independence

* Education vs marital status - is there a relationship?
* We will discuss statistical analysis of a sample of size n cross-classified in a  table with I rows and J columns. Such a configuration is called a **contingency table**.  The joint distribution of the counts $n_{i j}$ , where i = 1,..., I and j = 1,..., J , is  multinomial with cell probabilities denoted as $\pi_{i }j$ . Let $\begin{array}{l}{\pi_{i .}=\sum_{j=1}^{J} \pi_{i j}},\; {\pi_{. j}=\sum_{i=1}^{I} \pi_{i j}}\end{array}$, denote the marginal probabilities that an observation will fall in the ith row and  jth column, respectively.
* If rows and columns are independent of each other, then $\pi_{ij} = \pi_{i.}\pi_{.j}$. Therefore the null hypoth is $H_) = \pi_{ij} = \pi_{i.}\pi_{.j}$ for all i,j versus the alternative that the $\pi_{ij}$ are free. Under null, mle is $\begin{aligned} \hat{\pi}_{i j} &=\hat{\pi}_{i .} \hat{\pi}_{. j} =\frac{n_{i .}}{n} \times \frac{n . j}{n} \end{aligned}$. Under the alternative, mle is $\tilde{\pi}_{i j}=\frac{n_{i j}}{n}$. 
* Turning to Pearson test, $E_{i j}=n \hat{\pi}_{i j}=\frac{n_{i .} n_{. j}}{n}$, so $X^{2}=\sum_{i=1}^{I} \sum_{j=1}^{J} \frac{\left(n_{i j}-n_{i .} n_{. j} / n\right)^{2}}{n_{i .} n_{. j} / n}$, df = (I-1)(J-1)
* The chi-square statistic used here to test independence is identical in form and  degrees of freedom to that used in the preceding section to test homogeneity; however, the hypotheses are different and the sampling schemes are different. The test  of homogeneity was derived under the assumption that the column (or row) margins  were fixed, and the test of independence was derived under the assumption that only  the grand total was fixed. Independence can be thought of as  homogeneity of conditional distributions; for example, if education level and marital  status are independent, then the conditional probabilities of marital status given educational level are homogeneous

##### Matched-Pairs Designs

* The assumption behind the chi-square test of homogeneity  is that independent multinomial samples are compared, and sibling  samples are not independent, because siblings are paired.
* The null hypothesis is that probabilities of outcomes are the same for the X component and Y component of the pair in each sample, ie $\pi_{1 .}=\pi_{.1}$ and $\pi_{2.}=\pi_{.2}$ The null can be written as $H_{0}: \pi_{12}=\pi_{21}$. The off diagonal probabilities are equal and under the alternative they are not.
* McNemer’s Test: Under null MLEs of cell probabilities are $\begin{aligned} \hat{\pi}_{11} &=\frac{n_{11}}{n} ,\; \hat{\pi}_{22} =\frac{n_{22}}{n},\; \hat{\pi}_{12}=\hat{\pi}_{21} =\frac{n_{12}+n_{21}}{2 n} \end{aligned}$
* The $n_{11},\;n_{22}$ contributions to chi square test are 0 - we do not care about the diagonals
* $\begin{aligned} X^{2} &=\frac{\left[n_{12}-\left(n_{12}+n_{21}\right) / 2\right]^{2}}{\left(n_{12}+n_{21} / 2\right.}+\frac{\left[n_{21}-\left(n_{12}+n_{21}\right) / 2\right]^{2}}{\left(n_{12}+n_{21}\right) / 2} =\frac{\left(n_{12}-n_{21}\right)^{2}}{n_{12}+n_{21}} \end{aligned}$ with 1 degree of freedom

##### Odds Ratios

* $\operatorname{odds}(A)=\frac{P(A)}{1-P(A)}$ implying $P(A)=\frac{\operatorname{odds}(A)}{1+\operatorname{odds}(A)}$
* Now suppose that X denotes the event that an individual is exposed to a potentially  harmful agent and that D denotes the event that the individual becomes diseased. We  denote the complementary events as $\bar{X}$ and $\bar{D}$.
  * $\operatorname{odds}(D | X)=\frac{P(D | X)}{1-P(D | X)}$ and $\operatorname{odds}(D | \bar{X})=\frac{P(D | \bar{X})}{1-P(D | \bar{X})}$
  * The odds ratio is then $\Delta=\frac{\operatorname{odds}(D | X)}{\operatorname{odds}(D | \bar{X})}$ and measures the influence of exposure on subsequent disease
* Odds ratio can be the product of diagonal probabilities in the table divided by the product of the off diagonal probabilities: $\Delta=\frac{\pi_{11} \pi_{00}}{\pi_{01} \pi_{10}}$
* Prospective study: a fixed number of  exposed and nonexposed individuals are sampled, and the incidences of disease in  those two groups are compared. We can calculate the odds ratio but not the individual probabilities $\pi_{ij}$ since the marginal counts have been fixed by the sample design
* Retrospective study:  a fixed  number of diseased and undiseased individuals are sampled and the incidences of exposure in the two groups are compared. The joint / conditional probabilities cannot be calculated, but we can say $\operatorname{odds}(X | D)=\frac{\pi_{11}}{\pi_{01}}$, $\operatorname{odds}(X | \bar{D})=\frac{\pi_{10}}{\pi_{00}}$ leading to estimated odds ratio $\hat{\Delta}=\frac{n_{00} n_{11}}{n_{01} n_{10}}$