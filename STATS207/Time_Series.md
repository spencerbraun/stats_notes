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
* Parametric gives very accurate estimates when model assumptions are correct and allows for straightforward forecasting. But selecting the correct model may be difficult and the parametric form may be unrealistic in practice

##### Non-Parametric

* Smoothing / filtering - estimate the trend $m_t$ by averaging in a neighborhood: $[t-q, t+q]$. The window size needs to be chosen as a tuning parameter (here q).
* $\hat{m}_{t}=\frac{1}{2 q+1} \sum_{j=-q}^{q} x_{t+j}=\frac{1}{2 q+1} \sum_{j=-q}^{q} m_{t+j}+\frac{1}{2 q+1} \sum_{j=-q}^{s} Z_{t+j}$. When q is small then we have a very local estimate of trend, but the noise term is quite large - low bias high variance. With large q, we increase the bias but the noise term shrinks to 0. For example, if $m_t$ linear, then q can be quite large since we are estimating a constant slope - the first term is exactly equal to $m_t$ and we have no bias. Of course we could use a parametric model, but might be helpful if our trend looks piecewise linear smoothing makes some sense.
* There are many smoothing methods - exponential weighting, polynomial weights, kernel smoothers, etc. 
* Can decide whether to use symmetric window or only use past values - can think of forecasting financial time series, may only want a window using past values.
* Downsides - smoothing parameters can be difficult to choose optimally, no estimates for end points, no straightforward method for predicting future values - similar to issues with working with ML algorithms like random forests.

##### Isotonic Trend Estimation

* Less used than either of the other methods. 
* If expect a monotone trend, we solve a convex optimization problem. Min $\sum_{t=1}^{n}\left(X_{t}-a_{t}\right)^{2}$ st $a_{1} \leq \cdots \leq a_{n}$
* Fitting a piecewise linear function, allow the function to choose the knots where the slopes change. The monotonic restriction can be unrealistic but perhaps for shorter time periods. Has nice theoretical properties
* Not helpful if data is already monotone: If we are fitting to a monotonically increasing dataset, then we get back the original dataset - 0 residuals. We have not smoothed or learned anything.
* Unclear how to produce forecasts with this method.

### Differencing

* When $m_t \approx$ linear, then $Y_{t}=X_{t}-X_{t-1}=m_{t}-m_{t-1}+Z_{t}-Z_{t-1}$. For higher order trends, we can difference multiple times.
* Differencing when we want to remove a trend but do not need to estimate it.
* Linear Example: For example for $m_t = at$ for constant a. Let $x_{t}=a t+z_{t}$, then $\Delta x_{t}=x_{t}-x_{t-1} = \left(a t+z_{t}\right)-a(t-1)-z_{t-1} = a+z_{t}-z_{t-1}$ - constant mean, no trend!
* Quadratic Example: For $X_t = t^2$, note no noise, $\Delta X_{t}=X_{t}-X_{t-1}=t^{2}-(t-1)^{2} = 2t - 1$. There is stil trend remaining since we started with a quadratic. Instead $\Delta^{2} x_{t}=\Delta x_{t}-\Delta x_{t-1}=(2 t-1) - (2(t-1) - 1) = 2$, now no trend remaining. Continue for higher order trends and differences.
* Generally do not know order in advance - perform multiple times, compare ACFs to see which fits dataset best.
* In R, use `diff` for forward differencing, and `diffinv` can invert the differencing. Allows us to use differencing to estimate a model then generate un-differenced data from our model.
* Warning: differencing too many times can introduce autocorrlations. Starting with white noise and differencing, you can introduce signficance to ACF. Need to carefully difference, look at ACF, determine if white noise, try a higher order difference, determine what order is best. 

##### Prediction

* We will have to reverse the differencing. Say we have $\Delta x_{t}=x_{t}-x_{t-1}$. If $\Delta X_t$ has mean c for all t, then can forecast $\hat{X}_t = c + X_{t-1}$ by inverting our differencing to return to our original scale.
* For twice differenced $\Delta^2 X_t = \Delta\left(X_{t}-X_{t-1}\right) = X_t - 2X_{t-1} + X_{t - 2}$. If $\Delta^2 X_t$ has mean c for all t, then forecast is $\hat{X}_t = c + 2X_{t-1}  - X_{t-2}$
* We can forecast manually in R using this formula.

##### Stochastic Trend

* Random walk with drift, $m_t$ is a stochastic trend function, no longer deterministic 
* Popular model: $m_{t}=m_{t-1}+\delta+W_{t}$ 
* Differencing also works to remove trend in this case.

### Seasonality

* $X_{t}=s_{t}+Z_{t}$ where $s_t$ deterministic periodic function of known period d, $s_{s+d} = s_t$. Think of montly, quarterly, weekly data
* Fitting the model: superpositions of sin and cos terms: $s_{t}=a_{0}+\sum_{f=1}^{k}\left(a_{f} \cos (2 \pi f t / d)+b_{f} \sin (2 \pi f t / d)\right)$ with parameters $a_{0}, a_{1}, \dots, a_{k}, b_{1}, \dots, b_{k}$
* d is the longest period, so if we have monthly data with annual trends, d would be 12. Daily data with weekly pattern, d = 7. Since we define $s_{s+d} = s_t$, it is how long we wait in the data for the pattern to repeat.
* A small f is low frequency components, high f for high frequency. Usually start with low frequency and work upwards.
* d defines period st d / f = period. f / d = frequency, a = amplitude. 
* When we have sines and cosines, then period is least common multiple $\mathrm{LCM}\left(d / f_{1}, d / f_{2}\right)$. Take each individual period and take LCM. 
* Start by thinking what is the overall d of our data. Then start with the low frequency periods, which captures the largest repeated oscillations. Then consider adding higher freq compoenents, that will capture smaller repeated hills.
* Similar bias-variance tradeoff - how closely to we fit the high frequency components (increasing variance) or just the larger trends (higher bias)
* Non-Parametric Seasonal Function Estimation: $s_{i}:=\text { average of } \ldots, X_{i-2 d}, X_{i-d}, X_{i}, X_{i+d}, X_{i+2 d}, \dots$. Smooth across periods separated by overall period d.
* Seasonal differencing: $\nabla_{d} X_{t}=X_{t}-X_{t-d}=s_{t}-s_{t-d}+Z_{t}-Z_{t-d}=Z_{t}-Z_{t-d}$ - left with a mean 0 difference for all t. You will lose a full period d worth of observations in differencing.
* Trend and seasonality: $X_{t}=m_{t}+s_{t}+Z_{t}$ where m is a deterministic trend and s is a deterministic periodic function. 
  * Easy case is linear regression, smoothing / filtering a bit more challenging. We can also do differencing - first do seasonal differencing then do local differencing to get rid of the trend.
* Fitting frequencies to data - start by trying to fit the lowest frequencies and work upwords. Start with say 12/1, 12/2, 12/3...
  * In the beginning define the overall d (say 12), then work through frequencies, checking the fit using the acf. The correlogram may still look ugly and we need to correct for trend component, but hope to see seasonal cyclicality disappear from the plot.
* Smoothing with seasonality - say for quarters, want to weight the quarters equally, but that means if we have an odd window, we are giving double weights to some quarters. So want to underweight the quarters that are double counted. Otherwise will have remaining seasonality. Say estimating trend for quarter 2 - give 1/4 to quarters 1,2,3 then give 1/8 weight to q4 from prior year and 1/8 weight to q4 from current year - now we have symmetric weighting window while giving each quarter the same amount of weight.

### Variance Stabilizing Transform
* Often dealing with heteroscedastic data - say in stock data, the variance will change if the average price changes.
* Let $\operatorname{Var}\left(X_{t}\right)=g\left(\mu_{t}\right)$ for some known function g and mean of $X_t$, $\mu_t$. We are looking for a transformation $Y_t  = f(X_t)$ st the $Var(Y_t) = $ constant. Can be computationally difficult so tend to do a Taylor approximation
* Using $f(X_t) \approx f\left(\mu_{t}\right)+f^{\prime}\left(\mu_{t}\right)\left(X_{t}-\mu_{t}\right)$  - note $f\left(\mu_{t}\right)$ is deterministic since f, $\mu_t$ not random and just $\left(X_{t}-\mu_{t}\right)$ is random. Then $Var(Y_t) = Var(f(X_t)) \approx\left(f^{\prime}\left(\mu_{t}\right)\right)^{2} \operatorname{Var}\left(X_{t}\right)=\left(f^{\prime}\left(\mu_{t}\right)\right)^{2} g\left(\mu_{t}\right)$
* Goal is then to find f such that this variance approximation is constant.
* Example: $Var(X_t) = C \mu_t$. Want $f $ such that $(f^\prime(\mu_t))^2C\mu_t = $ constant. Try $f(x) = \sqrt{x};\; (\frac{1}{2\sqrt{x}})^2Cx = \frac{C}{4}$. Another option $f(x) = 2\sqrt{x} - 1$, etc these transformations are not unique. 
* Example: $Var(X_T) = C \mu_t^2$. Want $f $ such that $(f^\prime(\mu_t^2))^2C\mu_t^2 = $ constant. Here $f(x) = log(x)$ is a good pick: $(\frac{1}{\mu_t})^2C\mu_t^2 = C$ constant. 
* In practice - for count data, square root transformation is widely used. When the heterskedasticity does not look to be along a linear changing mean, then try the log transform.

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