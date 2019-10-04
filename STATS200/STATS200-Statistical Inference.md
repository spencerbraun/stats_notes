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

Convergence in Distribution

* $X_1, X_2$ sequence of RVs with CDFs $F_!, F_2...$. Let X be RV with distribution function F. SAy $X_n$ converges in distribution to X if:
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

