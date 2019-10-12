### Chapter 9 - Testing Hypotheses and Goodness of Fit

* 

##### Neyman-Pearson Paradigm

* Null hypothesis $H_0$ and alternative hypothesis $H_1$
  * Type I error: rejecting null when it is true. The probability of type I error is the significance level of the test $\alpha$
  * Type II error: accepting the null hypothesis when false, denoted by $\Beta$. 
  * Power: the probability that the null hypothesis is rejected when false, $1-\Beta$
  * Test statistic, statistic we are using as a test of the hypothesis
  * Null distribution: the probability distribution of the test statistic when the null hypothesis is true
* Suppose that $H_0$ and $H_1$ are simple hypotheses and that the test that rejects $H_0$  whenever the likelihood ratio is less than c and significance level $\alpha$. Then any other test for which the significance level is less than or equal toαhas power less than or equal to that of the likelihood ratio test.
* For simple hypothesis: We write down the likelihood ratio and observe that small values of it correspond in a  one-to-one manner with extreme values of a test statistic, in this case X. Knowing the null distribution of the test statistic makes it possible to choose a critical level that  produces a desired significance level $\alpha$.
*  If the alternative $H_1$ is composite, a test that is most powerful for every simple alternative  in $H_1$ is said to be uniformly most powerful. 
* In typical composite situations, there is no uniformly most powerful test. The alternatives $H_1 : \mu < \mu_0$ and $H_1 : \mu> \mu_0$ are called one-sided alternatives. The  alternative $H_1 : \mu = \mu_0$ is a two-sided alternative. 
* Confidence Intervals: $\mu_0$ lies in the confidence interval for $\mu$ if and only if the hypothesis test accepts. In  other words, the confidence interval consists precisely of all those values of $\mu_0$ for  which the null hypothesis $H_0: \mu= \mu_0$ is accepted.
* Suppose that for every value $\theta_0  \in \Theta$ there is a test at level $\alpha$ of the hypothesis  $H_0: \theta=\theta_0$. Denote the acceptance region of the test by $A(\theta_0)$. Then the set  $C(X) = {\theta: X \in A(\theta)}$  is a $100(1−\alpha)\%$ confidence region for $\theta$. Basically if a value for theta lies in the confidence region, the hypothesis test would be accepted for that value.
