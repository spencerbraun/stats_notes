---
title: Algorithm Analysis and Design
author: Spencer Braun
date: 20191005
---

Table of Contents

[TOC]

## Algorithms Part 1

### Asymptotic Run Time

* Adding two n digit integers is an O(n) operation since for each position we add at most three digits

##### Proving asymptotic run times

* Disproving Big O / Omega. Using contradiction show n can be compared to a constant. If $n_0$ is then set to be the constant, then $n > n_0$ will always be greater (less) than the constant, showing the contradiction.
* Proving Big O / Omega - simply need to show for some c and $n_0$ pair that we can bound the function. This works since we only need to show the inequality is true for some c, not all. 

### Divide and Conquer

##### Algorithm Design Considerations

* Divide the input into smaller subproblems, conquer the problems recursively, combine the solutions into a solution for the original problem
* Start by breaking the possibilities into cases - if we are trying to find this property, it could occur in the following ways. Given those ways, set up this portion as possible to do recursively, this portion needs to happen before or after recursion. Then think about run time - we are making every possible comparison, do we know of properties, facts, or goals that would reduce the amount of work we need to perform.
* MergeSort has the clever part hidden in the conquer portion, QuickSort has it hidden before the recursion, in deciding where to pivot. It can also occur in the recursive calls.
* When we are thinking about the size of the elements in an array, think how sorting could be used. When we are thinking about the index or ordering of an array that should not change, consider comparing certain values of specified indices. 
* After coming up with a brute force recursive algorithm, ask whether any work is duplicated or superfluous. Are there things we are doing that could eliminated like in Karatsuba or are unnecessary like in HW1.

##### The Master Method

* $T(n) \leq aT(\frac{n}{b}) + O(n^d)$
* a = number of recursive calls at an iteration of the algorithm (not all recursive calls total)
* b=input size shrinkage factor - how much smaller is the element passed to a recursive call
* d=exponent in running time for the comine step, since $O(n^d)$ is the work done outside of the recursive calls.
* Must be able to compose the algorithm in terms of this standard recurrence. a,b,d are constants, not dependent on n.
* $a = b^d \rightarrow O(n^d log \,n)$
* $a < b^d \rightarrow O(n^d)$
* $a > b^d \rightarrow O(n^{log_ba})$
* More general formulation: for $T(n) = aT(\frac{n}{b}) + f(n)$ for positive $a \geq 1$ and b > 1
  * if $f(n) = O(n^{log_ba-\epsilon})$ for some $\epsilon > 0$, $T(n) = \Theta(n^{log_ba})$
  * If $f(n) = \Theta(n^{log_ba}), T(n) = \Theta(n^{log_ba}logn)$
  * If $f(n) = \Omega(n^{log_ba+\epsilon})$ and if $af(\frac{n}{b}) \leq cf(n)$ for c < 1 and all sufficiently large n, then $T(n) = \Theta(f(n))$
* What this boils down to:  a is the number of subproblems at each recurrence. If we perform $n^d$ work at the first level, then each subsequent level performs $(\frac{n}{b})^d$ work. So the question is does the number of subproblems grow faster or slower than the work shrinkage factor of $b^d$? This is why a recursion tree can be helpful - sketch out how the subproblems spawn and the size of work shrinks.
* Altering recurrence relations to fit the formula: eg $T(n)=2T(\sqrt{n})+O(logn)$. 1) Try to find a substitution that might make this a more manageable relation. Here $k = log\,n$ seems reasonable so $n=2^k$. 2) Reformulate the RR using the new variable: $T(2^k)=2T(2^{k/2})+O(k)$. 3) Make a new RR s.t. we have a form $S(k)$ on the LHS. Here we use $S(k) = T(2^k) = 2S(\frac{k}{2}) + O(k)$. 4) Use the master theorem to find the runtime in terms of k. 5) Reverse the original substitution to get the runtime in terms of n.

##### Substitution Method

* Guess a function f(n) which you suspect satisties $O(n) \leq O(f(n))$. Prove by induction on n that this is true.
* Generally want to reverse engineer, if it looks like the recursion will be swamped by the outside factor, maybe use the constant factor as the  guess.
* Fix a positive integer $n \geq 2$ then try to prove $T(n) \leq l \times n$. 
* For the base case, we must show that you can pick some d s.t. $T(n_0) \leq d \times g(n_0)$ for our guessed function g(n) and d constant greater than zero. Then assume our guess is correct for everything smaller than n and prove it using the inductive hypothesis. Typically prove the inductive step from the hypothesis and obtain a condiiton for d.
* Inductive hypothesis we assume our guess is correct for any n < k and prove our guess for k.
* Concretely, we have $T(n) \leq aT(\frac{n}{n}) + O(n)$. Guess O(n log n), meaning prove  $T(n) \leq c n \,log\,n$ for an appopriate choice of c. Assume this holds for all positive values m < n, or m = n/2. Substitute m for n in the expression and simplify. May have many terms, but can try to bound with our original limit *given* a certain restriction on c (eg. c > 1). Prove with induction for $n > n_0$ where we get to choose $n_0$ to avoid tricky boundaries. Plug in constant values of n, see what values we get and pick a c s.t. the bound always holds. 
* If you have tricky expressions, bound the expression with a simpler one that eliminates low order terms.

##### Recursion Trees

* Start from level zero where there is no recursion. Split the tree down by the number of recursive calls / subproblems created at each level.
* The levels go from 0 to $log_bn$ where b is the shrinkage factor for the array size. At level j, each subproblem is size $\frac{n}{b^j}$ and the work performed is $c(\frac{n}{b^j})^d$. 

### Randomized Algorithms

* Probability concepts needed: linearity of expectation, indicator RVs, $\sum_{i=1}^{n-1} f(i) \leq \int_1^n f(x) dx$, expectations of Bernoulli, Binomial, and Geometric RVs.
* For QuickSort, if we always choose the first element on a sorted array, it runs in $O(n^2)$ time. It looks over every element comparing it to successive pivots. If the median were always the pivot, would run in $O(n log \, n)$ time since the recurrence relation is the same as MergeSort. A uniform random process runs in $O(n log \, n)$ shown through decomposition. Here the decomp shows that for element i and j to be compared, there are (j - i + 1) picks between them inclusive, 2 of which will compare them and the remaining in which they are split into different arrays and never compared. Therefore the expected number of comparisons is $\sum_{i=1}^{n-1}\sum_{j=i+1}^n\frac{2}{j - i +1}$ . This is then expanded as a power series, and bounded by the power series of ln(n). For the double sum, then bounded by 2n ln(n).
* For decomposition 1) identify the random variable Y 2) Express Y as a sum of indicator variables $X_1, ..., X_m$, $Y = \sum_{l=1}^m X_l$ 3) Apply linearity of expectation 4) Compute each $P(X_l=1)$ and add to find E(Y)
* Consider the running time of the algo of an input as a RV and bound the expectation of the RV

### Decision Trees

* Trying to demonstrate bounds of run time for all types of sorting models
* Compare two items at the top of three, then at the next step, can compare one of the same items to a new item. Transitivity tells us something about all 3 element. The leaves of the tree have the ordering of the elements compared by the upward branches. Internal nodes are boolean answers to comparison, leaf nodes correspond to outputs, all possible orderings of the items 
* Size of the tree depends on the size of the array. The elements in the array determine the path of the tree - running the algorithm corresponds to a single path in the tree, ie to the correct sorting of the inputs.

##### Proving Lower Bound for Comparison Sorting Models

* QuickSort takes a pivot, compares a single number to the pivot, and there is some indicator (say genie) that says whether the item is bigger that the pivot
* Theorem: comparison models are $\Omega(nlogn)$. Deterministic must take this many, randomized in expectation.
* Using a decision tree, we have every comparison down the tree ending in sorted output at the leaves. The runtime is the length of the path from the root to leaf for the correct output of the algorithm. Worst case run time would be the longest path from the root to any leaf.
* A binary tree has n! leaves - it contains every possible answer to the algorithm. Since it is a permutation of all items, this is n!. Then it must have $log(n!)$ depth at minimum, so it longest path must be at least log(n!). Then log(n!) = $\Theta(log(n^n)) = \Theta(nlog(n))$. 

### Proof Notes

##### Induction

* Induction - need to state we will be using induction, declare a base case, inductive hypothesis, inductive step, formally state the demonstrated results. The inductive hypothesis should be not quite the conclusion but the specific thing we will show in the inductive step. For example showing the first element is bigger than the last element, use A[0] > A[i] for any i, not jumping specifically to the first element is bigger than the last. 
* For your induction to be valid for arbitrary *n*, you must be able to point to a base case and sequence of inductive steps that show the claim extends to *n*. As an example, when *n*=3 using boilerplate induction, this corresponds to base case of 1, and inductive steps from 1 to 2 and 2 to 3. If you can formalize a way to show that each *n*from 1 to *k*can be expressed in this form, the induction is valid. 
* IH: After running for i iterations, we have achieved some goal expected after that iteration. 
* Base case: typically trivial, something 0, 1 to show our algorithm holds this 
* Inductive step: Know up to the i-1th iteration, want to show for i. Given generic variable inputs with certain invariant properties known after iteration i-1, explain how the algorithm would actually work upon those variables. This may involve laying out cases for different inputs. Once cases addressed, show that the algorithm does the right thing in all cases and returns an output as expected.

##### Contradiction

* Contradiction - clearly state proof by contradiction. This can be combined with induction if the inductive step is needed to show a contradiction. Always starts with assuming for the sake of argument the complement to the stated claim. 

##### Proving Correctness



### Glossary of Algorithms

##### Binary Search

* Divide and conquer to find a item in a sorted list. Look in the middle of the book and recurse on the left OR right depending on if value is greater or less than the mid point. 

##### Karatsuba

##### Matrix Multiplication

##### QuickSort

* Randomize algorithm for sorting. The random element is the choice of pivot. 1) Choose a pivot element, 2) rearrange the input array around the pivot s.t. all smaller elements go to the left, bigger to the right. Elements are only compared to the pivot, not to every other element. 
* Recursive calls occur after partitioning and there is no combine element.

##### RSelect

* ith order statistic is the ith smallest element in an array. Here we are trying to find that element for some specified i. 
* Choose pivot randomly, if we picked the ith order statistic done, otherwise determine if our pivot k > i look in the left portion of array, otherwise right half. If the median is always picked then the recurrence relation is $T(n) \leq T(\frac{n}{2}) + O(n)$ and it runs in $O(n)$. Worst case runs in $O(n^2)$. In expectation runs in linear time given properties of random pivots likely to be good pivots.

##### DSelect

* Instead of random pivot, pick pivot that is the median of medians. Divide list in to 5 parts, find the median of each part and find the median of the medians. The median of each sublist, we have a constant number of sublists and performs a constant amount of work per subarray, so computing the median takes constant time. Think of it as a MergeSort on the subarray of length n/5, leading to an O(n) routine.
* Has two recursive calls - one to find the median of medians, one to find the order statistic. The recursive calls play fundamentally different roles and can have different run times.
* Median of medians guarantees a split in the middle 40% of the array. Recurrence relation is then $T(n) \leq T(\frac{n}{5}) + T(\frac{7n}{10}) + O(n)$  which is roughly $T(n) \leq T(\frac{n}{5}) + T(\frac{7n}{10}) + cn$  since it is linear time outside of recursive calls.

##### BucketSort

* Uses the fact that we are actually sorting numbers. Have buckets to place numbers into from array, then just concatenate the the buckets. O(1) to put into buckets, then enumerated over each bucket O(n). Need to assume there are not too many values and need to know what values might show up ahead of time. 

##### RadixSort

* For sorting integers up to size M. Idea: BucketSort repeated, start by looking at the least significant digits, BucketSort, then repeat with the next least significant. Only need buckets 0-9. We place the full number in the bucket, just are sorting by a single digit at a time. Performed with linked lists because we need a FIFO system to get a sorted array at the end, iterating over the original and mid arrays from left to right.
* Given n d-digit numbers (in base 10), d iterations, O(n) per iteration, total is O(nd). d turns out to be $log_{10}n + 1$ so leads to O(nlog(n)), same as comparative algorithms.
* We need to change the base of the log. Bigger base means more buckets but fewer digits and fewer iterations. If r becomes very big, then you have BucketSort. For base 100 you have 100 buckets, 00-99.
* For n integers, max size M, and base r: # iterations = # digits, base r; d = $log_r(M) + 1$. Time per iteration is still O(n+r).
* Reasonable choice is r = n. -> $O(n \times(log_M(n) + 1))$. If $M \leq n^c$ for constant c, then this becomes O(n). If M is huge, this is not going to be a good algorithm.

## Data Structures

### Operations on Data

* Given some data structures like sorted arrays or linked lists, want to insert, delete
* In a sorted array, search is efficient in O(logn) but insert /delete mean moving things in memory, touching every element for O(n). In a linked list just replace the pointer for insert, but search / delete take a long time.

### Binary Tree

* Nodes without children really have NIL children
* Complete - every level (except last) is completely filled. In the last level all nodes are as far left as they can be.

##### Heaps

* Complete binary tree s.t. every descendent of a node has a larger key
* Insert fast and extract-min fast
* Insert - got to first empty node, plop new value there. Make sure we still have a complete binary tree, then make sure that the keys are sorted correctly for a heap. Then “bubble up” if the child is bigger than the parent node until they key is smaller than its children.
* Extract-min - Look at root, set min to top node. Bubble down to the smaller of the child keys, then re-sort down the tree to make a valid heap.
* In either case, swapping up the tree or down the tree with O(1) with each operation. In total, we have the O(height), and since it is a binary tree, this is O(logn).

##### Binary Search Tree

* Every left descendent of a node has key less than the node and right has key larger. Look up the tree to check if a tree is violating this property. You can make many different BST for a set of values.
* Similar to QuickSort - choose a root (pivot), then sort elements left and right, repeat.
* In order traversal of of BSTs outputs the elements in sorted order. Do the left first, print the key, then traverse the right. Runs in O(n).
* Search - traverse down the levels of the tree, O(logn). Insert - start with search, then insert when we cannot traverse to a closer key. Delete - search, delete, but also need to move children to new nodes, but this happens in O(1) time so the search time dominates.
* Could take O(n) if every node has a single child (poorly built). Instead of starting from scratch if it’s unbalanced, could use local operations to fix it. Turn to Red-Black trees - close enough to balanced that maintains itself

##### Red-Black Tree

* Every node is red or black. The root is black. NIL children count as black. The children of every red node are black. For all nodes X, all paths from X to NIL’s have the same number of black nodes (instead of trying to guarantee all paths have the same number of all nodes). Meaning for any NIL, you should have to get through the same number of black nodes when you pick the same starting place.
* This makes the black nodes completely balances and the red nodes need to be spread out. Easy to maintain these properties with insertion and deletion.
* The height of a red black tree with n nodes is at most 2log(n) - one side can only be double the short side by padding every other black node with red nodes.
  * Proof: IH: For subtree of size $\leq k$ (k nodes), $k \geq 2^{b(root)} - 1$
  * Base case: Tree of 1 node
  * Inductive step: ​k(x) = k(y) + k(z) + 1​
  * $k(x) \geq (2^{b(y)} - 1) + (2^{b(z)} - 1) + 1 \geq 2\times2^{b(x)(-1)} - 1 \geq 2^{b(x)} -1$
  * Notice we plug in the inductive hypothesis by plugging in for k(y), z and bounding
* Search only takes O(logn) since that is the height of the tree for sure. All other operations are O(logn) as well, though we won’t show exactly why.

## Probability Reference

* Binomial - n trials of a Bernoilli. Expectation = np
* Bernoulli - indicator with x=1 with probability p, x=0 with probability 1-p. Expectation = p.
* Geometric - number of trials until you see a success, where probability of success in each trial is p. Expectation = $\frac{1}{p}$