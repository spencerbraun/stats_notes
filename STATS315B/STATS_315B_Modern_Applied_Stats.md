---
author: Spencer Braun
date: 20200406
title: Modern Applied Statistics: Data Mining
---

[TOC]

# Modern Applied Statistics: Data Mining

## Introduction

* Boosting / additive trees very popular for unstructured data; now often most accurate for competitions and predictions
* Unobserved inputs Z that affect y in addition to X - why we cannot perfectly predict y. Instead we have a loss L. Alternatively can say $y = f(\textbf{x}) + \varepsilon$, for $\textbf{x} = (x_1,x_2,...,x_m)$. 
* $\varepsilon$ usually approximated as normal, though almost never is and usually not symmetric. There exist procedures to more accurately estimate its distribution if needed.
* Important to remembered squared error loss is not necessarily the most appropriate for a regression problem but is the easiest to work with. 
* Classification problem is simply assigning a name. There are no degrees of being wrong, either the right label assigned or it wasn't. Therefore the loss is a matrix of values, rows prediction, truth on the columns. All zeros on the diagonal since those are correct predictions, then losses elsewhere.
* Classification for assigning probabilities looks more like the regression problem since there are then degrees of wrongness.
* **Risk** $R(F)=E_{\mathbf{x} y} L(y, F(\mathbf{x}))$ - the average loss over future predictions. For each prediction I will lose something, and I want that to be as small as possible on average. 
* There will be a function that has the minimum prediction risk, and this is the predicting function we want to use. Optimal target function $F^{*}=\arg \min _{F} R(F)$. The target function, though it's the best possible, may not be very good - the unseen Z contributors may overwhelm our observed predictors.
* The risk $R(F)$ is over the joint distribution of x and y, over all the variables, but in order to take an expected value over the joint distribution we need to know this distribution. Since we do not know this distribution, we estimate it from a training sample where we have x and corresponding y.
* Concept drift - the distribution may change over time compared to our training sample, but we often pretend otherwise for our procedures.
* All linear procedures can be written as $\hat{F}(x)=\sum_{i=1}^{N} H\left(x, x_{i}\right) y_{i}$ where H does not depend on y and is a function of the x where the y value is and the value of x at which we want to make a prediction. Nearly all theory relies on fixed x's - meaning the new x's will look just like the one's we have in our training set, exact overlap of values.

### Choosing a Method

* If you average over all possibilities, no technique does better than any other. We just have information for our training data and the target function between points can be anything, then we have no way of knowing what it is. Have to impose assumptions and restrictions, say smoothness assumptions on top of other things.
* Do well if the real target function can be well estimated by a function in your restricted class of functions $F^*(x)$
* Each method has a situation where it works best. A **situation** is constituted by its real target function (unknown), training sample size (some methods have universality like NN or KNN methods as $n\rightarrow \infty$), signal / noise ratio (how much $\varepsilon$ compared to F, ie. s2n = $\frac{V(F(x))}{V(\varepsilon)}$). 
* With large noise, we need a more restricted procedure, while a NN might be great for high signal situation. Sample size is antidote to noise. 
* How do we choose? Try a number of them and focus on one performing best through CV. Try a committee - blend a variety of methods together, though of course won't work always. 

### Comparison Caution

* Empirical performance comparisons between methods should be taken with skepticism. Theory often lags new methods, so this is a common argument 
* There are no universal methods, consider the situations instead. 
* Selection biases - examples are selected to show the chosen method is best. Many situations may have been tried before they settled on this one, a type of overfitting. Paper selection effect - we are estimating how this performs, and it may vary but we only see papers when the method performs <u>best</u>
* Expert bias - the person performing the analysis is often more important than the method used. The more familiar you are with a technique, the more likely you can be powerful with it than another one. For a new procedure, the author has best idea for how to make it perform without incentive to tune the competitive methods as well.

### Machine Learning

* Every ML algorithm has 3 components
  * Model or pattern structure - the underlying functional form. Define a class of functions indexed by parameters, one of which we will use for our predictions. 
  * Score function - judges the quality of fit. We then try to maximize or min the score. The risk is the "magic" score that is the population score. We work with an approximation for a sample when we cannot derive this from first principles. Average loss function over the training data is a natural approximation, $\frac{1}{N} \sum_{i=1}^{N} L\left(y_{i}, F\left(\underline{x}_{i}\right)\right)$. But we almost always use another loss criterion, a close convex criterion that is easier to optimize.
  * Search strategy - find the function in that class that optimizes the score. Matrix algebra, quadratic programming, etc. Often have non-convex criteria and a direct solution is not possible - NN, clustering, trees. Heuristic search strategies introduce computational problems but also statistical difficulties. Increases the variance - with multiple minima, our sample matters quite a bit to the minimum we settle in as well as the starting point. Small changes in the starting point can make big changes to the solution I get.
* Searching in a function class that does not contain the true function - irreducible error or bias. In principle we could find the closest function in the class to the true function over the joint distribution. Instead we have a sample, and different samples will give us different scores. With a small function class, the bias can grow, since bigger classes are likely to be closer to the true function. But the bigger the function class, the harder it is to find the best function in that class - go to 0 bias, infinite variance.
* The biggest intellectual content now lives in the search strategy. It also drives our selection of the other two pieces - what is computationally feasible can select our score and model.
* In one dimension, a smoothness restriction might be a good enough class restriction. But in more dimensions we usually need to narrow our class further.
* The score criterion is the part that statistics deals with - it brings in the data. We need a training sample and to estimate the population score. 

### Data Basics

* Set of measurements each made on a set of N objects. $X_{ij}$ is the jth attribute on the ith object. This is matrix data, attribute-value data.
* Also can have proximity data - similarity or dissimilarity measure between objects, eg. networks, clustering. 
* The objects are observations and the measurements are variables. N = # of observations, n = # of variables.
* Predictor variables can take on many types
  * Interval scale variables - signed real numbers. They have (1) logically valued order relation and (2) real values continuous distance function. The most informative of all variable types.
  * Periodic / circular variables - direction, time of day, angle, longitude, months. No order relation, but there is a distance relation. Shortest distance around the circle that defines the period.
  * Ordinal variables - grades, size, performance, ratings. Order relation but no explicit distance function. Ranks are a subclass of ordinal variables - most informative ordinal, since as many distinct values as observations. Binary also a subclass, least informative ordinal but also the most flexible.
  * Categorical / nominal / factorial variables - words, places, languages, etc. Least informative of all variables types - no order relation and distance function can only be 1 or 0, same or different.
* Sometimes the context matters - color could be categorical for perception but interval scale in physics.
* Many procedures are designed specifically for interval scale variables - power weakened for other types, especially categorical. We impute distances between categories that do not reflect reality.
* Decision trees assume ordinal or categorical variables. Uses only the order relation, and for interval scale ignores the distances (loss of power). 
* Rule induction - treat all variables as categorical, ignoring distance and order. Lose a lot of information with this procedure for non-categorical predictors.
* Dummy encodings introduce correlations that don't exist - they have to sum to 1, so have pure collinearity. If one dummy is 1 then the rest are 0 - knowing one determines the other values.
* Contrasts - K-1 linear combinations of dummy variables - gets rid of the singularity introduced by using K.
* Outcome variable types
  * Interval scale -> regression problems. 
  * Periodic -> periodic regression. 
  * Ordinal outcomes -> ordinal regression. A separate literature in statistics
  * Binary -> binary classification / regression 
  * Categorical -> multinomial regression or multi-class classification. 
* Variable name type: When var name is categorical -> unstructured data. Structured data - where the name of the variable itself conveys information, eg time series, mass spectrum, image pixel values - the location, time, etc conveys information itself in addition to the value associated with it. This class deals with unstructured data.

## Chapter 9: Trees and Related Methods

* Decision trees can tell you exactly how it arrived at its decision - the opposite of a black box.
* Structural model: linear combination $F(\underline{x})=\sum_{m=1}^{M} c_{m} I\left(\underline{x} \in R_{m}\right)$, breaking up regions into spaces. There is a coefficient c for each region, if x in that region it gets valued with the constant c.
* Score criterion - looking for the regions and coefficients st we minimize the squared error from the true y values - least squares.
* Search strategy: given a set of regions, getting the coefficients is trivial. We can simply use the regions in a linear regression, a simple least squares problem (see score criterion). But optimizing wrt regions is a very difficult problem - therefore we will have to limit the class of regions we can look for, this is limiting our function class.
* Restrictions
  * We restrict the regions to be disjoint and the regions must cover the input space $\implies$ the regions are a partition of the input space. This then means that each x can be in only a single region. When a new x observation is entered, it will certainly be in a certain region and we can make a prediction for the new x. 
  * Simple regions - take the jth predictor variable, set of all values that $x_j$ can take on. Even if $x_1,x_2$ can take on a smaller range of values jointly than the square that bounds them, we restrict our regions to those rectangles.
* For categorical values, harder to define a subset of values in a range since no order relation. Then must explicitly delineate the subset of values, eg $s_j = \{a,b,c\}$
* Even a simplified problem is NP hard. We instead approximate with a greedy recursive partitioning. Initialize with full space S and the first region value is the global mean of y. At the mth iteration $F_{M}(\underline{x})=\sum_{m=1}^{M} \bar{y}_{m} I\left(\underline{x} \in R_{m}\right)$. We choose one of the regions and partition it into two child regions, giving us $F_{M+1}(\underline{x})=\sum_{m=1}^{M+1} \bar{y}_{m} I\left(\underline{x} \in R_{m}\right)$ and we can again choose a region to split, now also considering the child regions.
* Have to decide both which region to split next and where to split it. Since the regions are disjoint, we can look at the best split within a given region without worrying about the global space. Ultimately, we find our next region derived as $m^{*}=\operatorname{argmax}_{1 \leq m \leq M} \frac{N_{m}^{(l)} N_{m}^{(r)}}{N_{m}}\left(\bar{y}_{m}^{(l)}-\bar{y}_{m}^{(r)}\right)^{2}$. The fraction makes sure we partition such that we divide the number of observations more equally to make the split mean something.
* Where to split region, we look at just the data in our considered region, and consider a subset of values within our region along a given predictor. Consider a split at these values and consider how it affects our squared error loss. I'll pass over all regions, within a region pass over all variables, consider all possible splits and pick the region ($m^*$), variable ($j^*$), and split point ($t_{j^*m^*}$) that is optimal among our choices. 
* If our predictor is orderable numeric, can perform an exhaustive search for every value since finite. If our predictor is interval scale, we don't have to consider infinite values, only points that would change where a datapoint would fall in region. But our choice could still affect future data, so we simply choose the middle of the two datapoints since we have no information about this future data.
* Categorical variables - no order relation. We need to consider all combinations of splitting our categories into left and right. This grows exponentially with the number of levels. We have a trick approximation instead: consider the average y for each of the levels of a given categorical variable, then rank them on the means. We can then treat the variable as ordinal and run the algorithm normally. 
* Sensitivity: probability of predicting disease given true state is disease. Recall, true positive rate. Number of true predicted positives over all ground truth positives: $\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$
* Specificity: probability of predicting non-disease given true state is non-disease. True negative rate = $\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}$
* Precision: $\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}$.  Fraction of retrieved documents that are relevant to the query

