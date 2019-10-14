---
title: Algorithm: Examples and Strategies
date: 20191013
author: Spencer Braun
---

## Divide and Conquer

### Maximum Sum Subarray

* Brute Force: Calculate every subarray starting with the first element and moving to each subsequent element. To calculate all subarrays takes $O(n^2)$ $(n + n-1 + n-1...)$. Then finding the max is times O(n) giving us $O(n^3)$
* Key idea: 3 cases: max array can exist on left, right or middle. Check the middle before splitting and recursing.

1. Check the midpoint subarray moving outwards. O(n)
2. Split left and right and recurse over each side. O(log(n))
3. Return the max of the subarrays calculated. Left with O(nlog(n)) total, $T(n) = 2T(\frac{n}{2}) + O(n)$

### Lightbulbs and Sockets

* Basic idea: Sorting two lists relative to each other. 
* Brute Force: For each lightbulb, try every socket. $O(n^2)$
* Key idea: QuickSort but with with two arrays - one of lightbulbs and one of sockets.

1. For one lightbulb try every socket. Keep list of sockets too large and too small and the correct match. $O(n)$
2. Compare every lightbulb to to the matched socket. Keep list of too large and too small. $O(n)$
3. Recurse on list of small sockets and small lightbulbs / big sockets and big lightbulbs. If we pick a good lightbulb pivot, $O(log(n))$ levels.
4. Expected total runtime is then $O(nlog(n))$, the same as QuickSort

### Collinear Lines on a Plane

* Idea: Points on a plane, want to find m, b s.t. the maximum number of points possible lies on that line.
* Brute force: Take every pair possible, compute line parameters in $O(n^2)$ . Sort and find the maximum with the same parameters, total $O(n^2log(n^2)) = O(n^2log(n))$
* Given a limitation on the input: Maximum number of collinear points is $\frac{n}{k}$ for constant k.
* Key idea: The limiting factor allows us to do something otherwise time expensive. In expectation it is a fast approach

1. Pick pair of points randomly and compute m and b. Check to see if every other point lies on this line. $O(n)$
2. If fewer than $\frac{n}{k}$ points lie on the line, repeat. 
3. The probability of selecting two points on the line ie. the number of iterations: $\frac{n/{k} \choose 2}{n \choose 2}$ or $\frac{n/k}{n} \times \frac{(n-1)/(k-1)}{n-1} = \frac{1}{k^2}\frac{n-k}{n-1} \leq \frac{1}{k^2} = O(k^{-2})$ 
4. This is a geometric RV with p = $\frac{1}{k^2}$, therefore in expectation = $k^2$ until our first success. Not a function of n so combined $O(k^2n) = O(n)$. Note that it is not guaranteed to terminate.

### Catch Friend’s Lie

* Idea: Given a claim that contradicts the invariant, find an instance of contradiction. 
* Key idea: By checking the invariant holds, we ensure a contradiction exists within subarrays. Therefore we do not need to check each element and this is a binary search algorithm that follows a single recursion.

1. Check for invariant 1st element is less than last. If length < 2: return contradiction
2. Recurse on left or right if invariant holds until contradiction is found. $O(log(n))$

### Local Minimum in Array

* Idea: Find a single instance of an element whose neighbors are both larger or is an endpoint with one larger neighbor.
* Brute Force: go down array until find a local minimum making constant number of comparisons for each element. $O(n)$
* Key Idea: Finding one example so only need to recurse down a single path. Ensure invariant holds before we discard information.

1. From midpoint compare to two neighbors. 
2. If not a local min, recurse on one side with a smaller neighbor.
3. Repeat until at a local min or endpoint

### Local Minimum in a Grid

* Idea: Find a single instance of an element whose neighbors on all sides are larger or is an endpoint
* Brute Force: go down grid until find a local minimum making constant number of comparisons for each element. $O(n^2)$
* Key idea: Binary search on the rows, allows us to cut down matrix 1 half at a time. Trick is maintain invariant across entire row we are discarding.

1. Find minimum value in middle row. $O(n)$
2. Look above and below. If a value is smaller, recurse on that half of the matrix rows.

### Information Theory - Numbats

* Idea: Have a majority of truth tellers and minority liars. Want to find a single truth teller then all truth tellers.
* Brute force: Conduct votes on individuals until you find a majority vote for a truth teller. They can then identify all others. Worst case: $O(n^2)$
* Key Idea: We can make pairwise comparisons and given the majority, we can maintain the invariant. 

1. Make n/2 comparisons. Take all Truth-Truth pairs and take on individual from each. At most n/2 of them and guarantees majority truth tellers. $O(n)$
   
2. Recurse on subpopulation until have a single truth teller. $O(log(n))$
3. Total search is $O(n)$, since work at each level is shrinking
4. Note: even vs odd matters here and both cases need to be addressed. Here if we have an odd number of pairs TT, then we do not need the unpaired individual since we are assured a truth majority. If we have an even number of TT pairs, then we add in the individual to ensure truth majority (the individual is truth by the invariant).