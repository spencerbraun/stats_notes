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

##### Functions of RVs

* If $X∼N(\mu, \sigma^2) \text{and} Y = aX + b,\text{ then }Y∼N(a\mu + b, a^2\sigma^2 ).$
* **Change of variables for function of RV** $f_Y(y) = f_X(g^{-1}(y))\Big|\frac{d}{dy}g^{-1}(y)\Big|$
* Universality of uniform: Z = F(X) for $Z \sim U(0,1)$. $X = F^{-1}(U)$ then $F(x) = P(U \leq F(x))$

### Chapter 3: Joint Distributions

* Marginal frequency: $p_X(x) = \sum_i p(x, y_i)$
* Joint density, $f(x, y)$, then $F(x, y) = \int_{-\infty}^x \int_{-\infty}^y f(u,v) dv\,du$
* Marginal cdf $F_X(x) = \int_{-\infty}^x \int_{-\infty}^\infty f(u,y) dy\,du$, marginal density $f_X(x) = \int_{-\infty}^\infty f(x, y) dy$
* 

### Chapter 4: Expected Values

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
* The distribution of (n−1)S 2 /σ 2 is the chi-square distribution with n−1 degrees  of freedom. 
* Important: $\frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}$

