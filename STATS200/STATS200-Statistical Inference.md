---
title: Statistical Inference Notes
date: 20191003
author: Spencer Braun
---

**Chapter 5: Limit Theorems**

Weak Law of Large Numbers (Convergence in probability)

* $E(X_i) = \mu $ and $ Var(X_i) =\sigma^2$ . Let $X_n = n^{−1} \sum_{i=1}^n X_i$
* $P(|\bar{X_n} - \mu| > \epsilon) \rightarrow 0$ as $n \rightarrow \infty$
* $X_1 ... X_i \sim iid$
* Need to have  constant variance to apply WLLN -> otherwise must normalize across $X_i$s

Convergence in Distribution

* $X_1, X_2$ sequence of RVs with CDFs $F_!, F_2...$. Let X be RV with distribution function F. Say $X_n$ converges in distribution to X if:
* $\lim_{n\to \infty} F_n(x) = F(x)$ at every point for continuous F
* MGFs usually used to show this property. 

Continuity Theorem

* $F_n$ be a sequence of CDFs with MGF $M_n$. F CDF with MGF M. 
* If $M_n(t) \rightarrow M(t)$ for all t then $F_n(x) \rightarrow F(X)$ for all points of F

Central Limit Thm

* $X_1, X_2$ sequence of RVs with mean 0 and variance $\sigma^2$ common CDF F and MGF M.
* $S_n = \Sigma_{i=1}^n X_i$ 
* $\lim_{n\to \infty} P(\frac{S_n}{\sigma \sqrt(n)} \leq x) = \Phi(x)$ on $-\infty < x < \infty$



**Chapter 6: Distributions Derived from the Normal**

Chi Square Distribution

* U = $Z^2$ for $Z \sim N(0,1)$ , then $U \sim \chi^2_1$ 
* Note that the distribution is equivalent to $\chi^2_1 \sim \Gamma(\frac{n}{2}, \frac{1}{2})$
* If U1, U2, ... ,Un are independent chi-square random variables with 1 degree of  freedom, the distribution of V = U1 + U2 + ··· + Un is called the chi-square distribution with n degrees of freedom and is denoted by $\chi^2_n$

t-distribution

* $Z \sim N(0,1)$ and  $U \sim \chi^2_n$ for Z, U independent, then 
* $\frac{Z}{\sqrt{\frac{U}{n}}}$ is a t-distribution with n degrees of freedom
* density function of $$f(t) = \frac{\Gamma[\frac{1}{2}(n+1)]}{\sqrt{n\pi}\Gamma(n/2)}\Big(1 + \frac{t^2}{n}\Big)^{-\frac{n+1}{2}}$$

F Distribution

* U, V independent chi-square RVs wit m and n respective DoF
* $W = \frac{U/M}{V/n}$ is F with m and n DoF, ie. $F_{m,n}$

Sample Statistics

* For $X_1, ..., X_n$ iid normals, we sometimes refer to them as a sample from a normal distribution.
* $\bar{X} = \frac{1}{n} \sum_{i=1}^n X_i$ = sample mean
* $S^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2$ = sample variance
* Note that $E(\bar{X}) = \mu$ and $Var(\bar{X}) = \frac{\sigma^2}{n}$
* The RV X-bar and the vector of RVs $(X_1 - \bar{X}, ...)$ are independent
* $\bar{X}$ and $S^2$ are independently distributed
* The distribution of (n−1)S 2 /σ 2 is the chi-square distribution with n−1 degrees  of freedom. 
* Important: $\frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}$

