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

### Classification

* Outcome y takes on 1 of k non-orderable values, simply names. 
* Structural model will be quite similar to regression - partition space into regions, all observations in a region receive the same prediction. $\hat{c}(\underline{x})=\sum_{m=1}^{M} \hat{c}_{m} I\left(\underline{x} \in R_{m}\right)$
* Score criterion: prediction risk is still the master score criterion, $E_{y \underline{x}} L(y, \hat{c}(\underline{x}))$, expected misclassification cost. The difference here is that L is a matrix, K x K, the 2,7 entry is the loss when you classify as 2 and it is really a 7. L has 0's along diagonal and off diagonal elements provide the misclassification costs. In two class problem $\begin{bmatrix} 0 & c_1 \\ c_2 & 0\end{bmatrix}$
  * Can use misclassification error, treat all misclassifications the same. This is rarely true in real life - the costs for some misclassifications are much higher than others.
  * Data score criterion, $\hat{r} = \frac{1}{N} \sum_{i=1}^{N} L\left(y_{i}, c\left(\underline{x}_{i}\right)\right)$ natural score criterion 
* Search strategy - just like in regression, consider how a partition improves the prediction risk, recursive splitting + pruning.
* Again, the critical problem is finding a good set of regions, then assigning the class is trivial.
* For categorical X's, recall we need to use value subsets instead of split points.
* For entire tree $\hat{r}_M = \sum_{m=1}^M \hat{p}_m \hat{r}_m$ the estimated risk is the sum over regions of the probability of being in that region p times the estimated risk for that region, gives expected loss for that tree. $\hat{p}_{m}=N_{m} / N$ and $\hat{r}_m = \frac{1}{N_{m}} \sum_{\underline{x}_{i} \in R_{m}} L\left(y_{i}, \hat{c}_{m}\right)$, which is the risk given $\underline{x} \in R_m$. 
* Improvement then is $I(j_m, s_{jm}) = \hat{p}_{m} \hat{r}_{m}-\hat{p}_{l m} \hat{r}_{l m}-\hat{p}_{r m} \hat{r}_{r m}$. Seems like we have everything we need, but this works terribly! 
  * If I do a split into two children, I cannot improve my prediction risk if the children have the same predicted class with 0-1 loss. 
  * Example dataset has majority class 1's for every split, even though 3 splits could get us 0 error rate. But when looking for a first split, we cannot find a split where the children nodes will have different predictions, since class 1 is majority on both sides.
  * Misclassification risk is not a continuous function of model parameters. We can only do combinatorial optimization, for larger trees this is intractable
  * Each split minimizes estimated risk assuming it is the final split, does not account for better partitioning in future splits. Bad greedy strategy
* If we use a differentiable surrogate criterion, we can still use our greedy strategy. 
* Define the loss $\operatorname{loss}\left(y=c_{l}, c(\underline{x})=k\right) = \sum_{l=1}^{K}L_{lk}I\left(y=c_{l} | \underline{x}\right) = L_{lk}$ for when we predict k and the true class is $c_l$. 
  * Risk (the <u>expected</u> loss) in predicting $c(x) = c_k$ when truth is $y = c_l$ is $r_k(x) = EL(y=c_l, c(x) = c_k) = \sum_{l=1}^{K} L_{lk}E I\left(y=c_{l} | \underline{x}\right) =\sum_{l=1}^{K} L_{lk}Pr\left(y=c_{l} | \underline{x}\right) $ - the expected value of an indicator of an event is its probability that the event happens. We are summing over the classes 1 through k. If I classify as class k, the loss is the sum over the classes of the $L_{lk}$ times the probability that it really is an $\ell$. If I know the probability of y taking on each of its values at x, I can compute the optimal risk - this is the **Bayes** optimal decision rule.
* We do not know these probabilities, but we will estimate them as $\hat{p}_l(x)$. This changes our categorical problem to a interval scale problem. Then using asymptotic arguments can justify the use of our estimates. Our probabilities will be off with finite data, but our tree can still be optimal if the probability orderings, and therefore decision boundaries, are the same.
* We redefine our population score criterion: $E_{y \underline{x}} L\left(y, c(\underline{x})) \leftarrow E_{y \underline{x}} J\left(y,\left\{\hat{p}_{k}(\underline{x})\}_1^k\right.\right.\right.)$. Now we need an empirical score criterion that is differentiable and matches this expectation 
* Squared error loss is a candidate, where we estimate k numeric target functions, one for each class. Estimating the probability of each class at each value of x. We now have k variables we are trying to predict that are numeric, instead of 1 variable to predict that has k distinct values. We are using the probabilities as a device to do a minimum risk classification, but often the probabilities themselves are useful.
* Given a region, we simply estimate the probabilities by counting the number of observations in each class in the region: $\hat{p}_{km} = \frac{N_{km}}{N_m}$. However we do not just classify to the highest probability - the loss also depends on $L_{lk}$ for each comparison of class, if some misclassification are much more costly, we still might predict a lower probability class. 
* When we plug in, the squared error risk reduces to $\frac{N_{m}}{N} \sum_{k=1}^{K} \hat{p}_{km}\left(1-\hat{p}_{k m}\right)$, which is the Gini index of diversity.
  * Max diversity (min purity) when all classes equally probable, $G = 1 - \frac{1}{k}$.
  * Min diversity (max purity) when all 1 class, $G = 0$
* If the node is pure, it contributes nothing to the risk of the tree. If the node is not pure, it contributes a positive amount to the risk. Therefore our objective becomes making pure nodes. The improvement from a split becomes $\hat{I}_m(j , s_{jm}) = \hat{e}_{m}-\hat{e}_{m l}-\hat{e}_{mr} = \hat{P}\left(\underline{x} \in R_{m}\right) \text { Gini }\left(R_{m}\right) - \hat{P}\left(\underline{x} \in R_{m}^{(\ell)}\right) \text { Gini }\left(R_{m}^{(\ell)}\right) - \hat{P}\left(\underline{x} \in R_{m}^{(r)}\right) \text { Gini }\left(R_{m}^{(r)}\right)$
* This is differentiable - we are no longer trying to classify better, we are trying to purify. Splits that improve the confidence of your classification will be made, even if it does nothing to improve the misclassification error.
* Entropy: $H=-\sum_{k=1}^{K} P_k \log p_{k}$ also a fine criterion / purity measure, similar to Gini.

### Tree Advantages

* Relatively fast, can use all types of predictor variables - numeric, binary, categorical, missing values
* Invariant under monotone transforms of the predictor variables - building a piecewise constant model so order matters but scale does not.
  * Thus we do not have the issue of choosing transformations, simplifying the search process (eg in regression, we can make a <u>lot</u> of transformations)
  * Immunity to outliers in predictors. As long as the observations remain in the same order, the decision boundary is the same. Especially important for outliers that are typical in a single predictor dimension but are odd in 2 predictor dimensions or more - we cannot see them and we do not have to worry when using trees
  * Scales are irrelevant, allowing us to pull in data from different sources without worrying about rescaling
* Note we are not talking about outliers / transformation in y - those can matter since that determines the predicted value assigned
* Resistence to noise variables - we saw lasso, SVM etc degrade with high number of noise variables. Trees really do not require variable selection - the tree itself tells you what is important. Bet on sparsity principle: “Use a procedure that does well in sparse problems, since no procedure does well in dense problems.” Some people use trees for variable selection and place those into another model 
* Few tunable parameters - essentially off the shelf. Interpretable model representation. 
* No problems with highly correlated variables. The identifiability problem. The tree will just pick one a move on, so may hurt interpretation a bit but the predictive model should be fine.

### Tree Disadvantages

* Inaccuracy - piecewise constant approximation can lead to big bias.  Think of the linear target function, requires many splits to get arbitrarily close, but in practice you do not get that many splits on each variable. So piecewise constant functions do not always map well to the target - more bias.
* Hyper rectangular regions - oscillating overshooting and undershooting a linear function - bias issue
* Data fragmentation - each split reduced the training data in a subregion. Run out of data pretty quickly. Cannot model local dependency on more than a few variables, not good for target functions that have dependencies on many variables. Again another bias problem.
* Variance problems - we have to find a function in our function class and to do that we use data. Trees have very high variance caused by its greedy search strategy that results in a local optimum. $\hat{f}(x) = c\prod I(x_{j(l)} \in s_{jl})$ - we are multiplying our errors, causing the to get much larger!  Small changes in data cause big changes in the tree - unstable, high variance, high error.
* Look to bagging, boosting, and MARS to solve these issues.

## Chapter 8: Model Inference and Averaging

### Bagging

* Goal: improve performance of unstable procedures by stabilizing them. Ie, high variance procedures with multiple optima
* Given a convex criterion, our sample may have different minima. We get a distribution of solutions over samples, and this is our variance. Suppose instead we have a non-convex multiple minima criterion - this is our situation with trees, NNs. The solution we end up in will depend on where we start, which is not true in our convex case.  Procedures that are convex in their parameters are much more stable than those that are not. 
*  We have some $\hat{F}(x) = argmin \frac{1}{N} \sum_{i=1}^{N} L\left(y_{i}, F(\underline{x}_i)\right)$. Unstable procedure - small change in training data, big change in F hat.
* Bagging repeatedly makes small changes in data and averages the results. The average should be much more stable
* Now have $\hat{F}_b(x)$ for each iteration of our model fit to perturbed data. Then our bagged estimate is $\hat{F}_{B}(\underline{x})=\frac{1}{B} \sum_{b=1}^{B} \hat{F}_{b}(\underline{x})$. Typically we do a bootstrap sampling procedure to get perturbed data, but the procedure is not especially important.
* Here we aren't using the bootstrap for its original purpose of estimating population parameters. Just a convenient way to shake up the data, so other procedures could be just as effective. Dropout for NN's is equivalent to bagging - sampling from your parameters, averaging many solutions as the training goes along. We don't bag NNs since they take so long to train, use a concurrent regularization while training instead.
* Notice I have not changed my function space - bagging is pulling functions out of the same class. We do not reduce bias then, only change the variance. Why then not make the function class as big as we can, if we can reduce the variance with bagging? No reason - we do increase our function class and throw out pruning our pre-bagged trees!

### Random Forests

* Bagged trees where we randomize the available predictors to split over for a given tree fit. 
* Typically restrict to $\sqrt{p}$ where p is the number of predictors. Not especially chosen for theory but helps with computation.
* Increasing the randomization increases our variance control, but we pay with some bias.
* Bootstrap samples have a bias variance tradeoff - fitting on a portion of the dataset available introduces bias but we gain more from variance reduction. Fitting on the whole dataset lowers bias but we cannot control the variance.
* Work much better for classification than regression. This is true for trees in general, since classification is almost a piecewise constant function anyway.

## Chapter 10: Boosting and Additive Trees

### Boosting

* You can boost any procedure, but trees are especially improved by boosting. 
* Let $\left\{y_{i}, z_{i}\right\}_{1}^{N}$ be our outcome and predictor variables, $\hat{F}(\underline{z})=\operatorname{argmin}_{1} S\left(\left\{y_{i}, F\left(\underline{z}_{i}\right)\right\}_{1}^{N}\right)$ is our function approx in a function class, and S is our score criterion
* For trees, $F(z)=T(z)=\sum_{m=1}^{M} c_{m} I\left(z \in R_{m}\right)$. Then our boosted model is $F(z)=\sum_{j=1}^{J} a_{j} T_{j}(\underline{z})$, a linear model where we are finding the weights $a$ for each tree - it is a linear regression problem. This thing that defines our function in this space are the coefficients - we are considering all possible trees in our tree class (in principle). 
* We are not expanding our tree class like in bagging, we are weighting the models in our given class.
* Then our population optimization is $\left\{ a_{j}^{*}\right\}_{1}^{M}=\underset{\left\{a_{j}\right\}_{1}^{M}}{\operatorname{argmin}} E_{yz} L(y, \sum_{j=1}^{J} a_{j} T_{j}(\underline{z}))$ and on our training data approximated by $\hat{F}(\underline{z})=\arg \min _{\left\{ a_{j}\right\}_{1}^{M}} \frac{1}{N} \sum_{i=1}^{N} L\left(y_{i}, \sum_{j=1}^{J} a_{j} T_{j}\left(\underline{z}_{i}\right)\right)$
* Treat trees as fixed, weights are the parameters to solve for. So we can let $X_j = T_j(z),\; X_{ij} = T_j(z_i) \implies F(\underline{x} ;\{a_{j}^{*}\}_{1}^{J})=\sum_{j=1}^{n} a_{j}^{*} x_{j}$. 
* This parameter space is way too large - need to regularize to solve this optimization. We are essentially fitting a regularized linear regression. If N >> n, we will end up with a high variance answer - $\hat{R}$ is random and the optimization $\hat{\mathbf{a}}=\arg \min _{\mathbf{a}} \hat{R}(\mathbf{a})$ will vary widely.

### Regularization

* $\hat{R}(\mathbf{a})=\frac{1}{N} \sum_{i=1}^{N} L\left(y_{i}, a_{0}+\sum_{j=1}^{n} a_{j} x_{i j}\right)$ is the average loss over the data - empirical risk. With regularization we minimize this with a constraint $P(\mathrm{a}) \leq t$. Note $t \geq P(\hat{\mathbf{a}})$ imposes no constraint and max variance, t = 0 is max constraint and we have max bias (all coefs are 0)
* Define equivalent optimization problem $\hat{\mathbf{a}}(\lambda)=\arg \min _{\mathbf{a}}[\hat{R}(\mathbf{a})+\lambda \cdot P(\mathbf{a})]$. For a given data set, we have a fixed risk for a set of coefficients and the solution will change only with lambda. Given this set up, our set of possible solutions follows a 1 dimensional path $\in S^{n+1}$. We can then examine solutions along the path to find the best point - the one that minimizes the prediction risk.
* We cannot find the solution in the entire space of $a$, but our path restriction makes this problem tractable. This is different from a Lasso for example.
* If we were optimizing on finite data, lambda would be 0 - we obviously need to use CV. Construct the path using a subset of the data and use the left out data to approximate the predicted risk along the path.
* Make a grid of lambda values. For each lambda value we will solve our optimization problem. That gives us a set of coefficients, and we use the model given these coefficients to predict the left out data. That's how we choose lambda, but we still need a penalty function p. 
* We say $a^* = $ point in $S^{n+1}$. Different penalties will produce different paths in $a$ space. Want to use the penalty that brings us closest to the actual optimal solution target point. We then need to know something about the properties of the true solution. What is this property? Sparsity
* Sparsity: the fraction of non-influential variables. Even if we have measured 10k variables, assume only a few are actually influencing the outcome. We don't know which but we assume our solutions are sparse. In the case of trees, we pretty much need to assume sparse solutions - there are very few tress in our huge function class that will have good predictions on our outcome. Bet of sparsity principle.
* There are no true zero valued coefficients - with infinite data, each coefficient would have some weight, but we are forcing some to 0 for sparse solutions.
* Choose $P(a)$ that induces sparse paths - one that hugs one of the axes in $a$ space (say a has two dimensions). Keeps one axis near 0 and allows the other to have large influence. In higher space, hugs all axes but a few.
* **Power family**: $P_{\gamma}(\mathbf{a})=\sum_{j=1}^{n}\left|a_{j}\right|^{\gamma}$ indexed by gamma. This is quite familiar, $\gamma \in (2,1,0)$ produces ridge, lasso, and subset respectively going from densest to most sparse. Note we never penalize the intercept, just the slope parameters. With 2 highly correlated predictors, doesn't matter which we use in a prediction regression model - the data does not tell us which is "important", this is not a causal statement, merely the same for prediction's sake given the data. The L2 penalty will average the weight across these variables, while L0 would choose one with full weight and eliminate the other (L1 produces zeros, but generally less than L0).
* For L0, there is no penalty on coefficient size once the coefficient is non-zero. $(\hat{a}_0, \hat{\underline{a}}) = \underset{{a}_0, \underline{a}}{argmin} \; R({a}_0, \underline{a}) + \lambda \sum_{j=1}^m I(|a_j| > 0)$  -when $\lambda = \infty$ all a's set to 0, just intercept. Eventually reducing lambda reduced enough to point where risk reduction from non zero coefficient large enough to overwhelm the penalty. Eventually risk reduction will be greater than $2\lambda$ as we reduce the penalty - best 2 variable solution. Can continue to find best n-variable solution. The path will be piecewise - our penalty term is non differentiable and we have discrete jumps changing values of lambda. You get more discontinuities for smaller $\gamma$ : $\gamma = 0$ is simply most extreme with no continuous sections. Pure combinatoric optimization, and best 3 model may not contain best 2 model eg.
* Lasso is so popular because it is convex and produces sparse solutions - best of both worlds. Even if L1 and L0 pick coefficients in similar order, you will get different coefficient values - one has shrinkage and the other none. Elastic net produces denser solutions than L1 but still sets some coefficients exactly to 0.

##### Bridge Regression

* We need to solve repeatedly: $\hat{\mathbf{a}}_{\beta}(\lambda)=\arg \min _{\mathbf{a}}\left[\hat{R}(\mathbf{a})+\lambda \cdot P_{\beta}(\mathbf{a})\right]$ for $0 \leq \beta \leq 2,0 \leq \lambda \leq \infty$
* We look at penalties and point on the path constructed for that penalty that minimizes that criteria. Model selection criteria: $(\hat{\beta}, \hat{\lambda})$
* This is hard to construct with a fast algorithm, and very unreasonable for boosting or non-convex P. We could try ten different penalties and 100 different lambdas for each - this is not feasible.
* There are other ways for path construction: suppose we want $a^* = argmin_{\underline{a}} \hat{R}(\underline{a})$. Could use gradient descent - move towards lowest gradient, recompute, move, etc. We no longer need to solve the actual optimization at every point and could use our gradient path as our optimization path. 
* Early stopping - follow some descent path, each step making the risk smaller. Stopping short of the minimum often produces a better result - early stopping is a form of regularization. Before we get to $\lambda =0$ we find our optimum, but following the gradient isn't producing sparse solutions - closer to ridge. What we want is an optimization that would follow our lambda optimizing path, just much faster than the full optimization.
* **Direct path seeking**: rapidly produce path given P(a) such that the path is equivalent to the fully optimized solution.
  * Start with all coefs equal to 0 - solution for $\lambda = \infty$
  * At every step, we compute direction in coefficient space d and move a small amount $\Delta \nu$ in that direction
  * Update our location and repeat. until $\hat{R}$ is minimized
* Examples? Partial least squares, LARS 

### Generalized Path Seeking

* Generalized path seeking: fast for any convex loss function $L(y, F)$ and any penalty P(a) st $\frac{\partial P(\mathbf{a})}{\partial\left|a_{j}\right|} \geq 0$ - if you hold all the coef values the same and increase the value of one them, you increase the penalty - penalty is monotone increasing in the coefficient size for all coefs.
  * $\nu \geq 0$, point on path, orders the solution. $g_{j}(\nu)=-\left[\frac{\partial \hat{R}(\mathbf{a})}{\partial a_{j}}\right]_{\mathbf{a}=\hat{\mathbf{a}}(\nu)}$ - this is the negative gradient of the risk, derivative of the score function wrt each coefficient. $p_{j}(\nu)=\left[\frac{\partial P(\mathbf{a})}{\partial\left|a_{j}\right|}\right]_{\mathbf{a}=\hat{\mathbf{a}}(\nu)}$ - derivative of the penalty wrt the coefficient solutions.
  * $\lambda_{j}(\nu)=g_{j}(\nu) / p_{j}(\nu)$ - the jth component of the negative gradient divided by the jth component of the penalty at point $\nu$.  Lambda for each coefficient at each point on the path.
  * Then our algorithm: start with all coefs equal to zero, $\lambda = \infty$ solution
  * Compute lambdas at point $\nu$ using our penalty and gradient derivatives. Identify lambda values that have sign opposite of its corresponding coefficient (assume no sign for coef = 0). If that set is empty (and it usually is) then identify the coefficient j that has the largest value of lambda: $j^{*}=\arg \max _{j}\left|\lambda_{j}(\nu)\right|$. Otherwise, just look for this j in our opposite signed set.
  * Now we ID'd a single coef. Increment this coef by its estimated value $\hat{a}_{j^{*}}(\nu)$ at $\nu$ plus an increment $\Delta \nu$ times the sign of its $\lambda_{j^{*}}(\nu)$. Hold all other coefs to the same value. This gives us the next point on the path - $\nu \leftarrow \nu + \Delta \nu$
  * Repeat until $\lambda(\nu)=0$. 
* In a normal steepest descent, I would compute the gradient and move in that direction. Here taking a gradient ratio (if we are using Lasso penalty, then the penalty derivative is 1 for all coefs and we are actually using the gradient). And unlike normal steepest descent where we move in the gradient direction across covariates, here we move in the direction of a single predictor at a time, holding the others to their current value at a step. Picking the largest component of the gradient and only moving that direction. We do this bc it's not the destination that counts, it is the journey - the path is what matters to us. This path follows the optimized lambda path very closely. 
* Why do we expect this to follow closely the exact path? Say $\hat{\mathbf{a}}(\lambda)$ is the exact path and $\hat{\mathbf{a}}(\nu)$ is the GPS path - if the path as a function of lambda are monotone, then this algorithm produces the exact path. If as we relax lambda, the coefs either stay the same or increase, we get the exact optimized path. If not montone, at the point the coef begins to decrease (notice now the lambda value is the opposite of the sign of the coefficient since it is dragging it back towards zero), the GPS algorithms will stay constant for some period before decreasing and rejoining the exact path - creating an error distance between paths wherever the first derivative changes signs.