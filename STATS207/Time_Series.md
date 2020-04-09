---
title: Time Series
date: 20200407
author: Spencer Braun
---

[TOC]

# Time Series

## Introduction

* Time series can be modelled as stochastic process, typically handled in discrete time. 
* Trend - the secular movement over time vs seasonality, the repeated oscillations around the trend.
* Often perform transformations to get a more stable series, perform analysis, then transform back.
* Time domain: work directly with the data
* Frequency domain: work with Fourier transform
* IID noise: $X_t = W_t$ for $W_t \overset{iid}{\sim} f $ for some distribution f with mean zero and variance $\sigma^2 < \infty$. We have a special case of Gaussian noise, where $F = \Phi,\; \Phi(t) = \int_{-\infty}^t \frac{1}{\sqrt{2\pi}} e ^{\frac{x^2}{2}}dx$
* Harmonic oscillations plus noise: $X_t = \sum \alpha_k cos(\lambda_k t + \phi_k) + w_t$  
* Moving averages have short range dependence but no long range dependence, ie $Cov(X_t, X_{t-n_0}) = 0$. With AR, the covariance between any two terms is non zero. 
* Random walk $X_t = X_{t-1} + W_t$ without drift. Or we can add drift $X_t = X_{t-1} + 0.3 + W_t$
* Auto-regressive conditional heteroscedastic models ARCH: $X_t = \sqrt{r + 0.5x_{t-1}^2}W_t$ - variance changes with size of $|X_{t-1}|$
* Aliasing - distortion caused by sampling a continuous distribution into discrete at too low a frequency. Can change the appearance of the data completely

## Characteristics of Time Series

* White Noise
  * Sequence of RVs $X_i$ white noise if mean zero, finite variance and are uncorrelated
  * IID noise if white noise and iid 
  * Gaussian noise if they are iid noise and are normally distributed $X_{i} \sim \mathcal{N}\left(0, \sigma^{2}\right)$
* White noise has no varying structure over time - we need a test to see this. We cannot make any predictions from history of white noise. The standard approach will transform as given time series, fit a model to data, check whether residuals are white noise
* Moving Average: example, using immediate neighbors $v_{t}=\frac{1}{3}\left(w_{t-1}+w_{t}+w_{t+1}\right)$. The mean function does not change under a moving average transformation.
* A linear combination of values in a time series is referred to as a filtered series - using filter in R

### Measures of Dependence

* Autocorrelation: $\rho(s, t):=\frac{\operatorname{Cov}\left(X_{s}, X_{t}\right)}{\sqrt{\operatorname{Var}\left(X_{s}\right) \operatorname{Var}\left(X_{t}\right)}}=\frac{E\left(\left(X_{s}-E\left(X_{s}\right)\right)\left(X_{t}-E\left(X_{t}\right)\right)\right.}{\sqrt{E\left(\left(X_{s}-E\left(X_{s}\right)\right)^{2}\right) E\left(\left(X_{t}-E\left(X_{t}\right)\right)^{2}\right)}}$ for variables $X_i$ and lag k
  * It is the correlation between $X_s$ and $X_t$ and for white noise this should be zero
* Sample Autocorrelation Function: $r_{k}=\frac{\sum_{t=1}^{n-k}\left(X_{t}-\bar{X}\right)\left(X_{t + k}-\bar{X}\right)}{\sum_{t=1}^{n}\left(X_{t}-\bar{X}\right)^{2}}$ for $k = 1,2,...$
  * This is how we estimate the autocorrelation for a sample. There is some noise in esimating the autocorrelation, we rely on the LLN / CLT to make $r_{k} \approx \rho(k)=0$ for all $k \neq 0$
* Cross correlation function between two series, x and y: $\rho_{x y}(s, t)=\frac{\gamma_{x y}(s, t)}{\sqrt{\gamma_{x}(s, s) \gamma_{y}(t, t)}}$ for covariance function $\gamma$
* **Theorem A.7**: Under general conditions if $X_i$'s is white noise then for any fixed lag k and n large enough approximately $r_{1}, \ldots, r_{k} \stackrel{i . i . d .}{\sim} \mathcal{N}(0,1 / n)$. That is $\sqrt{n}\left(\begin{array}{c}
  r_{1} \\
  \vdots \\
  r_{k}
  \end{array}\right) \rightarrow \mathcal{N}(0, I)$ as $n \rightarrow \infty$
  * The correlogram - plot sample ACF's $r_1,..,r_k$ to check if white noise is a good model for the data
  * For white noise $\mathbf{P}\left(\left|r_{k}\right|>1.96 n^{-1 / 2}\right) \approx \mathbf{P}(|\mathcal{N}(0,1)|>1.96)=5 \%$ - a value of $r_k$ outside $\pm 1.96 n^{-1 / 2}$ is significant evidence against pure white noise - this is our white noise test. Be careful of multiple testing! If we have many $r_k$'s, say 10-20 lags, then the probability of one lag being signficant is high.
* Correlogram - First term is always 1, since the correlation of $X_t$ with itself = lag 0. Then each lag follows it with significance bands, with n = 100 we have $\frac{1.96}{\sqrt{100}} = 0.196 \approx 0.2$ so the bands are at $\pm 0.2$. 
* Multiple testing: what is the expected number of $r_k$'s exceeding the significance band under $H_0$? For example with 40 lags plotted, $E(\sum_{i=1}^{40} 1(|r_i| > 1.96/\sqrt{n})) = \sum_{i=1}^{40}  E(1(|r_i| > 1.96/\sqrt{n})) = \sum_{i=1}^{40} 0.5 = 2$. Even with white noise, we would expect to see 2 lags exceed the signficance bands on average. 

### Trend

* $m_t$ deterministic trend, $Z_t$ white noise. Then $X_t = m_t + Z_t$
* Estimate trend $\hat{m}_t$, remove it, then check if left with residual white noise. $X_t = \hat{m}_t \approx X_t - m_t = Z_t$
* Can be estimated with parametric form, smoothing / filtering, or isotonic trend estimation (convex optimization)
* Trying to use ML algorithms like random forests - it will give constant predictions outside of the training data. Does not continue the trend, simple takes the average of the last couple values. Does not see the time structure of the data, just a constant extrapolation from the last seen datapoints.
* Example: Parametric estimation via quadratic $\hat{m}_{t}=\hat{\alpha}+\hat{\beta} t+\hat{\gamma} t^{2}$. Objective find $(\hat{\alpha}, \hat{\beta}, \hat{\gamma}) \text { minimize } \sum_{t}\left(X_{t}-\alpha-\beta t-\gamma t^{2}\right)^{2}$
  * ACT on residuals - we have very few observations, so it looks like white noise but error bands are wide. Don't want to say white noise with confidence.
* CI's / PI's are not valid if the residuals are not white noise. Always look for white noise first before puting our model to use.

### Stationarity

* A **strictly stationary** time series is one for which the probabilistic behavior of every collection of values $\left\{x_{t_{1}}, x_{t_{2}}, \ldots, x_{t_{k}}\right\}$ is identical to that of the time shifted set $\left\{x_{t_{1}+h}, x_{t_{2}+h}, \ldots, x_{t_{k}+h}\right\}$, ie $\operatorname{Pr}\left\{x_{t_{1}} \leq c_{1}, \ldots, x_{t_{k}} \leq c_{k}\right\}=\operatorname{Pr}\left\{x_{t_{1}+h} \leq c_{1}, \ldots, x_{t_{k}+h} \leq c_{k}\right\}$

  * Note, for example, that a random walk process with drift is not strictly stationary because its mean function changes with time; a strictly stationary ts has a constant mean function over time
  * The autocovariance function of the process depends only on the time difference between s and t, and not on the actual times.
* A **weakly stationary** time series, $x_t$, is a finite variance process such that 

  * (i) the mean value function, $\mu_t$, is constant and does not depend on time t
  * (ii) the autocovariance function, $\gamma(s, t)$​ depends on s and t only through their difference |s − t|.
* Stationary means weakly stationary, since this is a more reasonable assumption for real data
  * Can simplify autocovariance to $\gamma(h)$ for h = s - t. The autocovariance function of a stationary series is symmetric around the origin.
* Stationary - white noise, three-point moving average
* Not stationary - random walk (AC depends on time), trend (mean dependent on time)