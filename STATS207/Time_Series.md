---
title: Time Series
date: 20200407
author: Spencer Braun
---

[TOC]

# Time Series

## Introduction

* Time series can be modeled as stochastic process, typically handled in discrete time. 
* Trend - the secular movement over time vs seasonality, the repeated oscillations around the trend.
* Often perform transformations to get a more stable series, perform analysis, then transform back.
* Time domain: work directly with the data
* Frequency domain: work with Fourier transform
* IID noise: $X_t = W_t$ for $W_t \overset{iid}{\sim} f $ for some distribution f with mean zero and variance $\sigma^2 < \infty$. We have a special case of Gaussian noise, where $F = \Phi,\; \Phi(t) = \int_{-\infty}^t \frac{1}{\sqrt{2\pi}} e ^{\frac{x^2}{2}}dx$
* Harmonic oscillations plus noise: $X_t = \sum \alpha_k cos(\lambda_k t + \phi_k) + w_t$  
* Moving averages have short range dependence but no long range dependence, ie $Cov(X_t, X_{t-n_0}) = 0$. With AR, the covariance between any two terms is non zero. 
* Random walk $X_t = X_{t-1} + W_t$ without drift. Or we can add drift $X_t = X_{t-1} + 0.3 + W_t$
* Auto-regressive conditional heteroskedastic models ARCH: $X_t = \sqrt{r + 0.5x_{t-1}^2}W_t$ - variance changes with size of $|X_{t-1}|$
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
  * This is how we estimate the autocorrelation for a sample. There is some noise in estimating the autocorrelation, we rely on the LLN / CLT to make $r_{k} \approx \rho(k)=0$ for all $k \neq 0$
* Cross correlation function between two series, x and y: $\rho_{x y}(s, t)=\frac{\gamma_{x y}(s, t)}{\sqrt{\gamma_{x}(s, s) \gamma_{y}(t, t)}}$ for covariance function $\gamma$
* **Theorem A.7**: Under general conditions if $X_i$'s is white noise then for any fixed lag k and n large enough approximately $r_{1}, \ldots, r_{k} \stackrel{i . i . d .}{\sim} \mathcal{N}(0,1 / n)$. That is $\sqrt{n}\left(\begin{array}{c}
  r_{1} \\
  \vdots \\
  r_{k}
  \end{array}\right) \rightarrow \mathcal{N}(0, I)$ as $n \rightarrow \infty$
  * The correlogram - plot sample ACF's $r_1,..,r_k$ to check if white noise is a good model for the data
  * For white noise $\mathbf{P}\left(\left|r_{k}\right|>1.96 n^{-1 / 2}\right) \approx \mathbf{P}(|\mathcal{N}(0,1)|>1.96)=5 \%$ - a value of $r_k$ outside $\pm 1.96 n^{-1 / 2}$ is significant evidence against pure white noise - this is our white noise test. Be careful of multiple testing! If we have many $r_k$'s, say 10-20 lags, then the probability of one lag being significant is high.
* Correlogram - First term is always 1, since the correlation of $X_t$ with itself = lag 0. Then each lag follows it with significance bands, with n = 100 we have $\frac{1.96}{\sqrt{100}} = 0.196 \approx 0.2$ so the bands are at $\pm 0.2$. 
* Multiple testing: what is the expected number of $r_k$'s exceeding the significance band under $H_0$? For example with 40 lags plotted, $E(\sum_{i=1}^{40} 1(|r_i| > 1.96/\sqrt{n})) = \sum_{i=1}^{40}  E(1(|r_i| > 1.96/\sqrt{n})) = \sum_{i=1}^{40} 0.5 = 2$. Even with white noise, we would expect to see 2 lags exceed the significance bands on average. 

### Trend

* $m_t$ deterministic trend, $Z_t$ white noise. Then $X_t = m_t + Z_t$
* Estimate trend $\hat{m}_t$, remove it, then check if left with residual white noise. $X_t = \hat{m}_t \approx X_t - m_t = Z_t$
* Can be estimated with parametric form, smoothing / filtering, or isotonic trend estimation (convex optimization)
* Trying to use ML algorithms like random forests - it will give constant predictions outside of the training data. Does not continue the trend, simple takes the average of the last couple values. Does not see the time structure of the data, just a constant extrapolation from the last seen data points.
* Example: Parametric estimation via quadratic $\hat{m}_{t}=\hat{\alpha}+\hat{\beta} t+\hat{\gamma} t^{2}$. Objective find $(\hat{\alpha}, \hat{\beta}, \hat{\gamma}) \text { minimize } \sum_{t}\left(X_{t}-\alpha-\beta t-\gamma t^{2}\right)^{2}$
  * ACT on residuals - we have very few observations, so it looks like white noise but error bands are wide. Don't want to say white noise with confidence.
* CI's / PI's are not valid if the residuals are not white noise. Always look for white noise first before putting our model to use.
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
* Quadratic Example: For $X_t = t^2$, note no noise, $\Delta X_{t}=X_{t}-X_{t-1}=t^{2}-(t-1)^{2} = 2t - 1$. There is still trend remaining since we started with a quadratic. Instead $\Delta^{2} x_{t}=\Delta x_{t}-\Delta x_{t-1}=(2 t-1) - (2(t-1) - 1) = 2$, now no trend remaining. Continue for higher order trends and differences.
* Generally do not know order in advance - perform multiple times, compare ACFs to see which fits dataset best.
* In R, use `diff` for forward differencing, and `diffinv` can invert the differencing. Allows us to use differencing to estimate a model then generate un-differenced data from our model.
* Warning: differencing too many times can introduce autocorrelations. Starting with white noise and differencing, you can introduce significance to ACF. Need to carefully difference, look at ACF, determine if white noise, try a higher order difference, determine what order is best. 

##### Prediction

* We will have to reverse the differencing. Say we have $\Delta x_{t}=x_{t}-x_{t-1}$. If $\Delta X_t$ has mean c for all t, then can forecast $\hat{X}_t = c + X_{t-1}$ by inverting our differencing to return to our original scale.
* For twice differenced $\Delta^2 X_t = \Delta\left(X_{t}-X_{t-1}\right) = X_t - 2X_{t-1} + X_{t - 2}$. If $\Delta^2 X_t$ has mean c for all t, then forecast is $\hat{X}_t = c + 2X_{t-1}  - X_{t-2}$
* We can forecast manually in R using this formula.

##### Stochastic Trend

* Random walk with drift, $m_t$ is a stochastic trend function, no longer deterministic 
* Popular model: $m_{t}=m_{t-1}+\delta+W_{t}$ 
* Differencing also works to remove trend in this case.

### Seasonality

* $X_{t}=s_{t}+Z_{t}$ where $s_t$ deterministic periodic function of known period d, $s_{s+d} = s_t$. Think of monthly, quarterly, weekly data
* Fitting the model: superpositions of sin and cos terms: $s_{t}=a_{0}+\sum_{f=1}^{k}\left(a_{f} \cos (2 \pi f t / d)+b_{f} \sin (2 \pi f t / d)\right)$ with parameters $a_{0}, a_{1}, \dots, a_{k}, b_{1}, \dots, b_{k}$
* d is the longest period, so if we have monthly data with annual trends, d would be 12. Daily data with weekly pattern, d = 7. Since we define $s_{s+d} = s_t$, it is how long we wait in the data for the pattern to repeat.
* A small f is low frequency components, high f for high frequency. Usually start with low frequency and work upwards.
* d defines period st d / f = period. f / d = frequency, a = amplitude. 
* When we have sines and cosines, then period is least common multiple $\mathrm{LCM}\left(d / f_{1}, d / f_{2}\right)$. Take each individual period and take LCM. 
* Start by thinking what is the overall d of our data. Then start with the low frequency periods, which captures the largest repeated oscillations. Then consider adding higher freq components, that will capture smaller repeated hills.
* Similar bias-variance tradeoff - how closely to we fit the high frequency components (increasing variance) or just the larger trends (higher bias)
* Non-Parametric Seasonal Function Estimation: $s_{i}:=\text { average of } \ldots, X_{i-2 d}, X_{i-d}, X_{i}, X_{i+d}, X_{i+2 d}, \dots$. Smooth across periods separated by overall period d.
* Seasonal differencing: $\nabla_{d} X_{t}=X_{t}-X_{t-d}=s_{t}-s_{t-d}+Z_{t}-Z_{t-d}=Z_{t}-Z_{t-d}$ - left with a mean 0 difference for all t. You will lose a full period d worth of observations in differencing.
* Trend and seasonality: $X_{t}=m_{t}+s_{t}+Z_{t}$ where m is a deterministic trend and s is a deterministic periodic function. 
  * Easy case is linear regression, smoothing / filtering a bit more challenging. We can also do differencing - first do seasonal differencing then do local differencing to get rid of the trend.
* Fitting frequencies to data - start by trying to fit the lowest frequencies and work upwords. Start with say 12/1, 12/2, 12/3...
  * In the beginning define the overall d (say 12), then work through frequencies, checking the fit using the acf. The correlogram may still look ugly and we need to correct for trend component, but hope to see seasonal cyclicality disappear from the plot.
* Smoothing with seasonality - say for quarters, want to weight the quarters equally, but that means if we have an odd window, we are giving double weights to some quarters. So want to underweight the quarters that are double counted. Otherwise will have remaining seasonality. Say estimating trend for quarter 2 - give 1/4 to quarters 1,2,3 then give 1/8 weight to q4 from prior year and 1/8 weight to q4 from current year - now we have symmetric weighting window while giving each quarter the same amount of weight.

### Variance Stabilizing Transform
* Often dealing with heteroskedastic data - say in stock data, the variance will change if the average price changes.
* Let $\operatorname{Var}\left(X_{t}\right)=g\left(\mu_{t}\right)$ for some known function g and mean of $X_t$, $\mu_t$. We are looking for a transformation $Y_t  = f(X_t)$ st the $Var(Y_t) = $ constant. Can be computationally difficult so tend to do a Taylor approximation
* Using $f(X_t) \approx f\left(\mu_{t}\right)+f^{\prime}\left(\mu_{t}\right)\left(X_{t}-\mu_{t}\right)$  - note $f\left(\mu_{t}\right)$ is deterministic since f, $\mu_t$ not random and just $\left(X_{t}-\mu_{t}\right)$ is random. Then $Var(Y_t) = Var(f(X_t)) \approx\left(f^{\prime}\left(\mu_{t}\right)\right)^{2} \operatorname{Var}\left(X_{t}\right)=\left(f^{\prime}\left(\mu_{t}\right)\right)^{2} g\left(\mu_{t}\right)$
* Goal is then to find f such that this variance approximation is constant.
* Example: $Var(X_t) = C \mu_t$. Want $f $ such that $(f^\prime(\mu_t))^2C\mu_t = $ constant. Try $f(x) = \sqrt{x};\; (\frac{1}{2\sqrt{x}})^2Cx = \frac{C}{4}$. Another option $f(x) = 2\sqrt{x} - 1$, etc these transformations are not unique. 
* Example: $Var(X_T) = C \mu_t^2$. Want $f $ such that $(f^\prime(\mu_t^2))^2C\mu_t^2 = $ constant. Here $f(x) = log(x)$ is a good pick: $(\frac{1}{\mu_t})^2C\mu_t^2 = C$ constant. 
* In practice - for count data, square root transformation is widely used. When the heterskedasticity does not look to be along a linear changing mean, then try the log transform.

### Stationarity

* Generally transform data for constant variance, remove trend components, then remove seasonality. Often we are not actually left with white noise, and this means we can make predictions based on past data.
* A **strictly stationary** time series is one for which the probabilistic behavior of every collection of values $\left\{x_{t_{1}}, x_{t_{2}}, \ldots, x_{t_{k}}\right\}$ is identical to that of the time shifted set $\left\{x_{t_{1}+h}, x_{t_{2}+h}, \ldots, x_{t_{k}+h}\right\}$, ie $\operatorname{Pr}\left\{x_{t_{1}} \leq c_{1}, \ldots, x_{t_{k}} \leq c_{k}\right\}=\operatorname{Pr}\left\{x_{t_{1}+h} \leq c_{1}, \ldots, x_{t_{k}+h} \leq c_{k}\right\}$
* The joint distribution of today's and tomorrow's random variables is the same as the joint distribution of the variables from any two successive days
  * Note, for example, that a random walk process with drift is not strictly stationary because its mean function changes with time; a strictly stationary st has a constant mean function over time
  * The autocovariance function of the process depends only on the time difference between s and t, and not on the actual times.
* A **weakly stationary** time series, $x_t$, is a finite variance process such that 

  * (i) the mean value function, $\mu_t$, is constant and does not depend on time t
  * (ii) the autocovariance function, $\gamma(s, t)$​ depends on s and t only through their difference |s − t|.
* Stationary means weakly stationary, since this is a more reasonable assumption for real data
  * Can simplify autocovariance to $\gamma(h)$ for h = s - t. The autocovariance function of a stationary series is symmetric around the origin.
  * The autocorrelation function no longer depends on t $\rho(h) \triangleq \frac{\gamma(h)}{\gamma(0)}$ for $\gamma(h)=\gamma(t, t+h)=\operatorname{Cov}\left(X_{t}, X_{t+h}\right)$
* Stationary - white noise, three-point moving average
* Not stationary - random walk (AC depends on time), trend (mean dependent on time)
* Can take two periods from the time series and plot their histograms - a stationary time series should have overlapping histograms of their values, while a non-stationary will have separated histograms.
* Can have a half gaussian, half chi-square white noise time series - will looks non stationary but by using distributions with the same moments, we still have a weakly stationary process. 
* For the family of Gaussian processes (joint distribution for any period is multivariate normal), strong and weak stationarity are the same. While generally strong implies weak, in this special case weak also implies strong.

## ARIMA Models

### Moving Average Models

* MA(1) = $X_{t}=Z_{t}+\theta Z_{t-1}$ for Z white noise. We are modeling using noise plus past noise.
* This is a weakly stationary process: Given a white noise process $Z_t,\; Var(Z_t = \sigma^2),\; \theta \in \R$. Set $X_{t}=Z_{t}+\theta Z_{t-1}$. Then $E(X_t) = 0,\; Var(X_t) =\sigma^2 + \theta^2\sigma^2$. Looking at covariance, $Cov(X_t, X_{t-1}) = Cov(Z_t+ \theta Z_{t-1}, Z_{t-1} + \theta Z_{t-2}) = \theta \sigma^2 \neq 0 $ but for longer lags $Cov(X_t, X_{t-j}) = 0 $ for $j > 1$.
* Generally a MA(q) process = $X_{t}=Z_{t}+\theta_{1} Z_{t-1}+\theta_{2} Z_{t-2}+\ldots+\theta_{q} Z_{t-q}$ for non zero $\theta$ parameters. $X_t$ is a linear combination of white noise contributions (not a real average, just a linear model). Every MA(q) model is weakly stationary.
* The autocovariance given by $\gamma_{X}(h)=\sigma^{2} \sum_{j=0}^{q-h} \theta_{j} \theta_{j+h}$ if $h=0,1, \ldots, q$ and 0 beyond lag q. The influence of lags cuts off at q. There is no t in this formula, so we have weak stationarity - covariance independent of time period.
* Looking at the autocorrelation function, for a lag larger than the order of the process it should have zero influence on $X_t$. We should look for this pattern in data - significant lags back to a certain q, then drop off to 0. This will be opposed to an AR process that exhibits exponential decay in the lags, without a single drop off lag value.
* Backshifting $BX_t = X_{t-1}$, so an MA(q) is equivalently written as $X_t = \theta(B) Z_t$ for $\theta(B)=1+\theta_{1} B+\ldots \theta_{q} B^{q}$. We are building towards a polynomial times a white noise term, allowing us to say more about the MA process.
* Invertibility: two MA models could explain the same dataset. 
  * For example, a Gaussian noise process could be described by $X_{t}=Z_{t}+\theta Z_{t-1} \simeq Z_{t-1}+\theta Z_{t}=\tilde{Z}_{t}+\frac{1}{\theta} \tilde{Z}_{t-1}$ for $\tilde{Z}_{t}=\theta Z_{t}$. These are two different MA models that generate the same time series.
  * Let's assume $Var(Z_t) = \sigma^2 = 1$, have $X_{t}=Z_{t}+\theta Z_{t-1}$. Then $E(X_t) = 0,\; Var(X_t) = 1 + \theta^2,\; Cov(X_t, X_{t-1}) = \theta, \; Cov(X_t, X_{t-j}) = 0 \forall j > 1$. Instead look at $\tilde{X}_t = \tilde{Z}_t + \frac{1}{\theta}\tilde{Z}_{t-1};\, Var(\tilde{Z}_t) = \theta^2$ and $E(\tilde{X}_t = 0),\; Var(\tilde{X}_t) = \theta^2 + 1$ and we get the same covariances as well -> distributions of $X_t,\;\tilde{X}_t$ are the same despite coming from different MA processes as $Z_t,\;\tilde{Z}_t$ are Gaussian white noise.
  * We have a natural fix: $|\theta| < 1$ , meaning the noise at time t has larger influence than noise at time t-1
    * $|\theta|<1 \iff \theta(z)=1+\theta z \neq 0 \;\forall|z| \leq 1$
* Definition: A MA(q) model is **invertible** if $\theta(z) \neq 0 \text { for }|z| \leq 1$ for $X_{t}=\theta(B) Z_{t}$. The roots of the backshift polynomial are less than or  equal to 1.
* Theorem: A model is invertible iff the white noise process can be written as a linear combination of the observations. 
  * Before we wrote the observed values as a linear model of the noise, now we invert to write the noise as a combination of the values. $Z_{t}=\pi(B) X_{t}=\sum_{j=0}^{\infty} \pi_{j} X_{t-j}  \Leftrightarrow  X_{t}=-\sum_{j \geq 1} \pi_{j} X_{t-j}+Z_{t}$ where $\pi(B)=\sum_{j=0}^{\infty} \pi_{j} B^{j} $ and $ \sum_{j=0}^{\infty}\left|\pi_{j}\right|<\infty $ and $ \pi_{0}=1$
  * The formulation in terms of X is quite useful for prediction. 
  * Intuition: can write the noise process as a stable function of the observed values.
* Example: usefulness of invertibility. 
	* Take a process with a non-invertible representation: $Z_t \sim^{iid} U(0,1)$ and $X_t = Z_t + 2 Z_{t-1}$. Trying to write this as a function of the past plus some noise: $X_t = Z_t + 2(Z_{t-1} + 2Z_{t-2}) - 4Z_{t-2} = Z_t + 2X_{t-1} - 4Z_{t-2}$. This can be iterated to get $X_t = Z_t + 2X_{t-1} - 4 X_{t-2} + 8 X_{t-3} - 16X_{t-4}...$. Why is this problematic? As we go further into the past the weights are exploding, the infinite past has an insanely large influence on your future predictions. 
	* Taking the same distribution in an invertible representation: $\tilde{Z}_t \sim^{iid} U(0,4)$ and $\tilde{X}_t  = \tilde{Z}_t + \frac{1}{2}\tilde{Z}_{t-1}$. Writing in terms of past values, $\tilde{X}_t  =  \tilde{Z}_t  + \frac{1}{2}(\tilde{Z}_{t-1}  + \frac{1}{2}\tilde{Z}_{t-2} ) - \frac{1}{4} \tilde{Z}_{t-2}  =  \tilde{Z}_t +  \frac{1}{2}\tilde{X}_{t-1}  - \frac{1}{4} \tilde{Z}_{t-2} $. Iterating: $\tilde{X}_t =  \tilde{Z}_t +  \frac{1}{2}\tilde{X}_{t-1}  - \frac{1}{4} \tilde{X}_{t-2} +  \frac{1}{8} \tilde{X}_{t-3}$. We now have decaying weights towards the past, which seems preferable. 
	* The same process has two representations in terms of past values + noise $Z_t,\; \tilde{Z}_t$. For practical reasons, we will focus on the stable representation -> invertible condition.
	* The point is that there are multiple valid parameter representations of the same process - not a unique solution. By restricting the parameter space we consider to invertible models, we are able to model data outside of the training support with some stability. Even if the non-invertible representation can accurately represent our current data, it is useless for prediction.

##### Checking for Invertibility

* Are the roots of the characteristic polynomial outside of the unit circle?
* Example: $X_t = Z_t - Z_{t-1} + \frac{1}{4}Z_{t-2}$
	* Invertibility only if the polynomial has roots outside of the unit circle 
	* $\theta(B) = 1 - B + \frac{1}{4}B^2$, since our model given by $X_t = \theta(B)Z_t =  (1 - B + \frac{1}{4}B^2)Z_t$
	* Condition for invertibility: $\theta(Z) \neq 0, \; \forall |Z| < 1$. Solving the quadratic equation, we get solutions = $\theta_{1/2} = \frac{1 \pm \sqrt{1-4(1/4)}}{2(1/4)} = 2$. So we can factorize the polynomial as $\theta(B) = (1- \frac{1}{2}B)^2$
	* We see the root is outside the unit circle, so the process is invertible. (Note we are looking at the characteristic polynomial of a matrix, think of taking a matrix to a power, equivalent to taking the eigenvalues to a power. In the limit, need the eigenvalue to be less than 1 to converge)
* Example: $X_t = Z_t - 3Z_{t-1} + 2Z_{t-2}$
	* $\theta(B) = 1 - 3B + 2B^2$. We find the roots of the polynomial are $\theta_{1/2} = \frac{3 \pm \sqrt{3^2 - 4\times2}}{4} = \frac{3 \pm 1}{4} = \{1,1/2\}$. The process is not-invertible, as one root is on the boundary of the unit circle. 

##### MA with Infinite Order
* Example: $Z_j \sim^{iid} N(0, \sigma^2); \; \theta_j = \phi^j,\;\phi \neq 0 $. Then $X_t = \sum_{j=0}^\infty \theta_j  Z_{t-j} =  \sum_{j=0}^\infty  \phi^jZ_{t-j}$. Pulling out the first noise component, $= Z_t +  \sum_{j=1}^\infty \phi^j Z_{t-j} = Z_t +  \phi \sum_{j=1}^\infty \phi^{j-1} Z_{t-j} = Z_t +  \phi \sum_{j=0}^\infty \phi^{j} Z_{t-1-j}$. So we pulled out the first term and reformulated to look like the original process for $X_{t-1}$. So we see $X_t = Z_t + \phi X_{t-1}$
* We have seen this process before, and it is simply an AR(1) model. In other words, we can use a MA process to encode AR processes and we can go back and forth. They are two notations for equal models.

### Autoregressive Models
* AR(p) model is of form $X_{t}=Z_{t}+\phi_{1} X_{t-1}+\phi_{2} X_{t-2}+\ldots+\phi_{p} X_{t-p}$
* Working from the infinite MA process, we can define the autocovariance and autocorrelation
* Autocovariance: $\gamma(h) = \sigma^2\sum_{j=0}^\infty \theta_k \theta_{j+h} = \sigma^2 \sum_{j=0} \phi^j \phi^{j+h} = \sigma^2 \phi^h \sum_{j=0}^\infty \phi^{2j}$. This is a geometric series, so we can rewrite as $ \sigma^2 \phi^h \sum_{j=0}^\infty (\phi^{2})^j = \sigma^2\phi^h \frac{1}{1-\phi^2}$
* Autocorrelation: $\phi(h) / \phi(0) = \phi^h$
	* Exponential decay of autocorrelations! Notice this is quite different from MA, where we had lags up to a point and then a drop off to zero at a particular lag
	* For $\phi > 0$ get trending down of positive lags. For $\phi < 0$, we get shrinking lags but with positive and negative autocorrelation oscillations. Seeing these oscillations in the ACF is a good indication we need an AR process.
* Formally the AR(p) model is of the form $X_{t}=Z_{t}+\phi_{1} X_{t-1}+\phi_{2} X_{t-2}+\ldots+\phi_{p} X_{t-p}$ for non zero $\phi$
	* $X_t$ is a linear combination of past values plus white noise. Clearly this will be useful for predictions, since this form directly gives us a formula for predicting future values.
* Autoregressive operator: $\phi(B)=1-\phi_{1} B-\ldots \phi_{p} B^{P}$. 
	* Recall for the MA process, we had $X_t = \theta(B)Z_t$. Here we have the opposite: $\phi(B) X_{t}=Z_{t}$ for white noise Z
* Unique stationary solution: For some white noise process Z and fixed parameter $|\phi| \neq 1$ there exists exactly one time series process X with mean zero which is stationary that solves the difference equation $X_{t}-\phi X_{t-1}=Z_{t}$. The solution is given by $X_{t}=\sum_{j \geq 0} \phi Z_{t-j}$
	* Other non-stationary solutions to $X_t = \sum_{j=0}^\infty \phi^j Z_{t-j}$: if we add a trend that scales with the growth component $X_t = \sum_{j=0}^\infty \phi^j Z_{t-j} + \alpha \phi^t$. For any deterministic or random $\alpha$ we still have a solution to $X_{t}-\phi X_{t-1}=Z_{t}$. So there are many non-stationary solutions to this recursion but only one stationary solution.
* Intuition: For polynomial $\phi(Z)  =1 - \phi Z$ can write the process in compact form $\phi(B) X_t = Z_t$. If $\phi(B)$ were a number, we could divide both sides by $\phi(B)$ to obtain $X_t = \frac{1}{\phi(B)}Z_t$, a unique representation of our process. We can just proceed by pretending it is a number $X_t = \frac{1}{1 - \phi B}Z_t$, we could use a geometric series. Then since $\frac{1}{1-X} = 1 + X + X^2+... $ if $|X| < 1$. Here $=(1 + \phi B + \phi^2 B^2 + ... ) Z_t$. So the guess based on this intuition that the unique stationary solution to $\phi(B) X_t = Z_t$ is $X_t = \sum_{j=0}^\infty \phi^j B^j Z_t$ 
  * This can be made more precise by using a Neumann Series, checking $|\phi B| <1$, but this requires functional analysis. B is just a shift matrix, shifts your process but doesn't change the variance, etc. So the norm of this operator is exactly 1, so $|\phi B| < 1\iff \phi < 1$, and this is equivalent to the unit root problems we looked at for MA process: $\phi < 1 \iff $ characteristic roots outside of the unit circle.
* Proof of Lemma for assuming $|\phi | < 1$ (as we will exclude other cases later): We NTS $X_t = \sum_{j=0}^\infty \phi^j B^j Z_t $ satisfies $X_t = \phi X_{t-1} + Z_t$. Then $X_t = \sum_{j=0}^\infty \phi^j B^j Z_t  = \sum_{j=0}^\infty \phi^jZ_{t-j} = Z_t + \phi X_{t-1}$. So our intuitive solution turns out to be correct. This approach (division -> geometric series) works quite often in converting between representations.
* Uniqueness: Let's assume there is another process $Y_t$ that satisfies $Y_t  = \phi Y_{t-1} + Z_t$. Then $Y_t = Z_t + \phi (\phi Y_{t-2} + Z_{t-1}) = Z_t + \phi Z_{t-1} \phi^2 Y_{t-2}$. Can continue to iterate this, $Z_t + \phi Z_{t-1} \phi^2 Z_{t-2} + ... + \phi^k Y_{t-k}$ where the last term cannot be converted to Z. If the remainder goes to 0, then we have the same process as above. $E[(\phi^k Y_{t-k})^2] = \phi^{2k} E(Y_{t-k}^2)  = \phi^{2k} E(Y_0^2)$ since we are looking at stationary solutions only, so constant in t. Then $\underset{k \rightarrow \infty}{lim} \phi^{2k} E(Y_0^2) = 0$ if $|\phi| < 1$. Therefore $Y_t = \sum_{j=0}^\infty \phi^j Z_{t-j}$ and thus $X_t = \sum_{j=0}^\infty \phi^j Z_{t-j}$ is the unique stationary solution.
	* Remark: $|\phi| > 1$ can be treated similarly, but the unique stationary solution will suddenly depend on the future values of the noise process, which is highly problematic. 
* If $\phi(z) = 0$ for some $|z| < 1$ then the stationary solution depends on future values of $Z_t$. This violates what we call a causality condition
* AR(p) model $\phi(B) X_t = Z_t$ is **causal** if $\phi(z) \neq 0$ for $|z| \leq 1$.
* AR(p) mopdel is causal iff the time series and white noise can be written as a stable function of past noise values. (The moving average representation $X_{t}=\psi(B) Z_{t}=\sum_{j=0}^\infty \psi_{j} Z_{t-j}$)
* Example: (non-causal process) $X_t = 2X_{t-1} + Z_t$. Want to write the process as a MA - could do this through recursion or the backshift operator + inversion, but latter needs invertibility. With recursion: 

$$
X_t = 2(2X_{t-2} + Z_{t-1}) + Z_t = Z_t + 2Z_{t-1} + 4X_{t-2} \\
= Z_t + 2Z_{t-1} + 4(2X_{t-3} + Z_{t-2}) \\
= Z_t + 2Z_{t-1} + 4Z_{t-2} + 8X_{t-3}
$$

* First problem we see is that the weights explode exponentially, and the variance of the remainder term $2^k X_{t+k}$ also explodes.
* Trying a different way: $X_t = 2X_{t-1} + Z_t \iff X_{t-1} = \frac{1}{2} X_t -  \frac{1}{2} Z_t$ A time reversed process with coefficient of 1/2 instead of 2 - now our coefficient will degrade instead of explode. We get 

$$
X_t = 2X_{t-1} + Z_t \iff X_{t-1} = \frac{1}{2} X_t -  \frac{1}{2} Z_t \\
= -\frac{1}{2}Z_t + \frac{1}{2}(\frac{1}{2}X_{t+1} - \frac{1}{2}Z_{t+1}) 
= -\frac{1}{2}Z_t - \frac{1}{4}Z_{t+1} + \frac{1}{4}X_{t+1} \\
=-\frac{1}{2}Z_t - \frac{1}{4}Z_{t+1} - \frac{1}{8}Z_{t+2}... + \left(\frac{1}{2}\right)^k X_{t+k}
$$

* Weights go zero exponentially fast and remainder goes to zero . So we can write $MA(\infty)$ representation as $X_{t-1}  = -\frac{1}{2}Z_t - \frac{1}{4}Z_{t+1} - \frac{1}{8}Z_{t+2}...$. It depends on future values, which is not useful for modeling the real world. The problem is $\phi(B) = 1-2B,\; \phi(1/2) = 0$ violates the unit root condition and we see we have a non-causal process.
* Other examples:
  * $X_t = 0.9X_{t-1} +Z_t \implies \phi(B) = 1- 0.9B$ so $\phi(Z) = 0 \iff Z = 10/9 > 1$ - yes this process is causal. (Note careful with the signs compared to MA process. If we had $X_t = Z_t +0.9Z_{t-1}$ MA process, then $\theta(B) = 1 + 0.9B$. For the AR process, have to move our X's to the same side before defining the polynomial, while the backshift polynomial works on the Z's for the MA process)
  * Random walk: $X_t  = X_{t-1} + Z_t \implies \phi(Z) = 1-Z$. We are exactly at the boundary of causality (boundary of unit circle. The roots $\phi(1) = 0 \implies $not causal. But this is widely used in finance - how do we fix this? Notice that this process is not stationary  - once we tranform to a stationary process we no longer have this problem. Differencing: $\Delta X_t = X_t - X_{t-1} = Z_t$. When you have a root very close to 1, differencing is commonly used to return the process to stationarity
  * $X_t = -X_{t-2} + Z_t;\; \phi(Z) = 1 + Z^2$. We then have complex roots $\phi(Z) = 0 \implies Z = \pm i$, which means this process is not causal.
* Notice for MA(q) process, invertible $(\theta(z) \neq 0 \text { for all }|z| \leq 1) \Leftrightarrow A R(\infty)$, want parameters uniquely identified. For AR(p) process, causal $(\phi(z) \neq 0 \text { for all }|z| \leq 1) \Leftrightarrow M A(\infty)$ - want unique stationary solution that only depends on the past.

### ARMA Models

* ARMA(p,q) is linear combination of MA(q) and AR(p) of form $\phi(B) X_{t}=\theta(B) Z_{t}$. We always assume that $\phi(z), \theta(z)$ have no common factors - this removes redundancies. Our models we have seen so far can be seen as special cases ARMA(0,0), ARMA(p,0), ARMA(0,q)
* Common factors in $\phi,\theta$: For example, $X_t= X_{t-1} + Z_t - Z_{t-1}$ implies polynomials $\theta(z) = 1 - z,\; \phi(z) = 1-z$ and common factor $1-z$. This looks like an ARMA(1,1) process. But one solution is $X_t = Z_t$, ie X is just white noise: $X_t = Z_{t-1} + Z_t - Z_{t-1} = Z_t$, so in fact this is ARMA(0,0). We can fit this model but end up with a complicated description of a very simple process. (Note we use z instead of B in the polynomial bc B is an operator whereas z is a complex number that solves the polynomial equation.)
* We are interested in the simplest model that fits the data. In practice start with lower order ARMA models before trying to build more complex ones
* Example: $\theta(z) = (1-\frac{1}{3} z)(1- \frac{1}{2} z),\; \phi(z) = (1-\frac{1}{2}z)(1- \frac{1}{4}z)$ They share one factor, so after removing common factors get $\theta(z) = (1-\frac{1}{3} z),\; \phi(z) =(1- \frac{1}{4}z)$. End up with ARMA(1,1) instead of ARMA(2,2).
* ARMA models have causal and invertible conditions on the polynomials for MA and AR as defined above. ARMA is causal if can be written as a $MA(\infty)$ representation (linear combination of noise), and invertible if can be written as $AR(\infty)$ representation.
* This requires dividing two polynomials: $\psi(z)=\theta(z) / \phi(z)$. We write $\phi(z) \psi(z)=\theta(z)$, could look at constant terms on each side, get formula for $\psi_0$, look at linear terms, etc. But instead can invert the operator and use geometric sums: $\phi(z)=\left(1-a_{1} z\right) \dots\left(1-a_{p} z\right)$ and write $\psi(0) = \theta(0)\left(1-a_{1} z\right) \dots\left(1-a_{p} z\right)$.
* Why are we interested in MA and AR representations? The MA representation tells us something about the ACF. The AR representation can be used for prediction. 
* Example: $X_t = \frac{1}{3} X_{t-1} + Z_t - \frac{1}{2}Z_{t-1};\; \phi(z) = 1 - \frac{1}{3} z;\;\theta(z) = 1 - \frac{1}{2}z$. Now use method 2 to get an MA represenation: $\frac{\theta(z)}{\phi(z)} = \frac{1 - \frac{1}{2}z}{1 - \frac{1}{3}}$ Then using geometric series, multiplying out and collecting by order terms: 

$$
=(1 - \frac{1}{2}z)(1 + \frac{1}{3}z + (\frac{1}{3}z)^2 + (\frac{1}{3}z)^3...) \\
= 1 + \frac{1}{3}z  + (\frac{1}{3}z)^2 + (\frac{1}{3}z)^3 ... -\frac{1}{2}z - \frac{1}{2}z\frac{1}{3}z - \frac{1}{2}z(\frac{1}{3}z)^2 \\
= 1 + \sum_{j=0}^\infty z^{j+1}\left(\left(\frac{1}{3}\right)^{j+1} -\frac{1}{2}\left(\frac{1}{3}\right)^j\right)
$$

* Thus we get the following MA representation $X_t = Z_t + \sum_{j=0}^\infty Z_{t-j-1} (\left(\frac{1}{3}\right)^{j+1} -\frac{1}{2}\left(\frac{1}{3}\right)^j )$
* In practice, R does this for us `ARMAtoMA`