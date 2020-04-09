---
author: Spencer Braun
date: 20200406
title: Modern Applied Statistics: Data Mining
---

[TOC]

# Modern Applied Statistics: Data Mining

## Introduction

* Boosting / additive trees very popular for unorganized data; now often most accurate for competitions and predictions
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
* Variable name type: When var name is categorical -> unorganized data. Organized data - where the name of the variable itself conveys information, eg time series, mass spectrum, image pixel values - the location, time, etc conveys information itself in addition to the value associated with it. This class deals with unorganized data.

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

### Tree Representation

* Split from parent region to children region corresponds to a parent node into children nodes
* Builds a binary tree that contains the same information as the regions. 
* With each internal node, we have a split, the subset of the associated values telling you to go the left or right. With categorical values we need to know which subset of values goes to each child node.
* Get a final set of regions, where all X's in a region share a predicted Y value, the mean of the associated Y values in the region.
* We don't have to tell trees what variables are important - it determines on its own which are important. If it is in the tree with our constraints, it is among the important variables. The tree also allows us to take an X and walk through the logic that leads to its eventual prediction value. Decision trees -> $\hat{F}(x)=\sum_{m=1}^{M} \bar{y}_{m} I\left(x \in R_{m}\right)$
* Decision trees give complete information for how a decision was determined. Could tell someone exactly why they were rejected for a loan say.
* Trees are a universal estimator - with enough data, it can approximate any underlying function.

### Missing Predictor Values

* If various predictor values are missing, trees have a simple way to handle them. 
* Would like the quality of the model to degrade gently as the number of missing values increases, not totally collapse given a trained model. We additionally would like to build trees using training sets with missing values
* Say we are in a region and some values of the predictor we are looking at are missing. We can perform surrogate splits as a local imputation
  * Take $X_j$ vs $X_k$ - say these predictors are highly correlated. If we have a certain split for $X_j$ from the model, the observed values go left or right (assign 1 or 0). Then for values of $X_k$, we essentially want to build a predictor that guesses whether the $X_j$ label would be 1 or 0. Then when values of $X_j$ are missing, we can use our predicted value from the observation's $X_k$ value.
  * Once we find that we have a missing value for a split, we pass over all the other variables for which we have values and use the one with the highest correlation with our missing predictor. This is the first surrogate. If the first surrogate is missing, we continue to the second surrogate, etc.
* Keep in mind this is being done <u>locally</u>. Once we have done some splitting and we are looking at correlations among segments of the predictors, we may find associations that we did not see before. A loopy convex curve could become linear once we have performed some splits.
* Sometimes missing could be informative - extreme ends of income spectrum might be missing, not at random. This procedure does not take this into account, we would want another procedure if that matters to us.

### Right Sized Trees

* Recall we are working with piecewise constant predictions. If working with a diagonal line, could make huge errors at the boundaries, could be fixed with more regions but eventually run out of data before we could split all variables.
* If we have too few regions, we have representational bias - our function is not represented by a piecewise constant. If we have too many splits, we are in danger of overfitting. If $y = f^*(X) + \varepsilon$, we train on y since this is what is observed. This means the $\varepsilon$'s are included in training, so if we train too far, we fit to the idiosyncratic error that will not generalize to other data.
* The index controlling the size of the function class for trees is the number of regions. The larger we expand our function class (regions) the more susceptible to noise we are. Adjusting the size of the function space is regularization generally, and here our tuning parameter is # of regions (# of terminal nodes). The bigger the function class, the harder it is to find the best function approximation within it - this is the bias variance tradeoff.
* Two region models are less flexible than 3 region models. The space of 3 region models contains all 2 region models
* Prediction risk: $risk \sim bias^2 + variance$. As tree gets larger bias down, variance up, and there is an optimal number of regions for a particular problem. We want to estimate that.
* Could simple bound the number of regions which would bound our variance. But some areas will need more splits in areas where function changes a lot, while others do not. We cannot address this with this strategy.
* Stopping rules: could determine if next split is not worthwhile. There is always some improvement with our next split (even if just due to $\varepsilon$). Let $\hat{e}_{m }=\frac{1}{N} \sum_{\underline{x}_{i} \in R_{m}}\left(y_{i}-\bar{y}_{m}\right)^{2}$ be the contribution of region m to the error from the truth. Then the improvement of another split to the squared error risk is $\hat{I}_{m}=\hat{e}_{m}-\hat{e}_{m L}-\hat{e}_{m R}$ which will always be positive - optimistically biased estimate of $I_m$. We could threshold this value, but there may be later splits that are actually quite significant that we would miss. This can be seen with a symmetric interaction effect between two predictors - the estimated Y value is the same on both sides with one split and leave error, but 2 splits would leave no error.
  * $\sum_{m=1}^M \hat{e}_m$ = total risk of the tree = total loss over the entire tree
  * There will always be improvement from another split due to the noise. We then would be fitting our model quite closely to idiosyncratic noise.
  * The threshold $\hat{I}_m \geq k$ is not a sufficient condition for optimal splits.
* Look ahead one level: Then $\hat{I}_{m}=\hat{e}_{m}-\hat{e}_{m+1}-\hat{e}_{m+3}-\hat{e}_{m+4}$ where m + {1,3,4} are the terminal nodes after 2 splits, with threshold $\hat{I}_{m}>{2} k$, then accept the two splits else make m terminal. If risk from those child terminal nodes is less than $\hat{e}_m$ then we may exceed our threshold.
* Full look ahead: Let $\hat{r}_{m} \equiv \hat{e}_{m}$ then for terminal m $C_{m}=\hat{r}_{m}+k$ else $C_{m}=\sum_{m^{\prime} \in m} C_{m^\prime}$ for terminal descendents. We charge a cost k for each terminal node. Make choice to minimize $C_m$. Termination rule can then be if $\hat{r}_{m}+k \leq C_{m_{L}}+C_{m_{R}}$ then make m terminal and set $C_{m}=\hat{r}_{m}+k$. Else accept the split and set $C_m= C_{m_{L}}+C_{m_{R}}$
  * The point is to observe this rule, we grow the largest possible tree and then apply this rule to determine the number of terminal nodes. We end up applying the rule in inverse order of depth. 
  * In practice, split until we are left with identical Y's in terminal nodes (then can split no more) - maximal variance tree. Look at the deepest split, and look at the contribution to risk from the two children compared to the parent. If the sum of the child risks + 2k less than the parent risk + k, then keep the split. 
  * Since we are splitting all the way down, we can implement it with recursion - we continue to go left until no more steps, backtrack then continue. In practice often only split until a certain min number of observations in a terminal node but this is a computational approximation - we can split until all nodes are pure.
* Choosing k - the bigger k, the smaller tree we will have, since it is the cost charged for each terminal node.
* All this is equivalent to an optimization problem, **cost complexity pruning**: T = set of all possible tree arbitrarily terminated. $t \in T$ and $|t|$ = number of terminal nodes = M. Our job is to select one of those trees. Then $\hat{r}(t) = \sum_{m=1}^{|t|}\hat{r}_m$ = empirical risk for tree t. Then we seek to find $t_{k}^{*}=\underset{t \in \mathcal{T}}{\operatorname{min}}[\hat{r}(t)+k|t|]$ for k = complexity parameter (penalty). Regularizing on k, not exactly the same as the number of terminal nodes. 
  * If we have two values of k, then the size of the tree for the larger k value will be less than or equal to the tree formed from the smaller k: $k^{\prime} > k \Rightarrow\left|T_{k^\prime}^{*}\right| \leqslant\left\lfloor T_{k}^{*}\right|$
  * Starting with k=0 and raising k. At some point, raising k reaches a point where 2k overwhelms k for the parent no matter what the comparitive risk is. Then the node collapses. We can keep raising k to collapse more nodes. We get a nested seqnece of trees indexed by k. Penalizes for increased variance associated with more complex model $\operatorname{Var} \hat{y} \sim |T| \sigma^{2} / \mathrm{N}$
  * Instead of exhaustive tree search, we are left with a sequence. Still left with how much to charge for each node k, but this will depend on the signal to noise ratio.
  * In high dimensional space, we really cannot just look and see whether we are fitting a signal better or noise better like we can in 2D. Instead we rely on CV.
* With small sample size, often introduce too much bias with a validation set. Then can turn to k-fold CV.

