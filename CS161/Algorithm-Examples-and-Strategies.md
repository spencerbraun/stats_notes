---
title: Algorithm: Examples and Strategies
date: 20191013
author: Spencer Braun
---

[TOC]

## Divide and Conquer

### Maximum Sum Subarray

* Brute Force: Calculate every subarray starting with the first element and moving to each subsequent element. To calculate all subarrays takes $O(n^2)$ $(n + n-1 + n-1...)$. Then finding the sum is times O(n) giving us $O(n^3)$
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

### Majority Element

- Idea: Array with $\frac{n}{2} +1$ elements with value x, can only make equality comparisons, want to find the majority element x.
- Brute Force: Run through array, comparing each value to every other until you find an element that is equal to a majority of the other elements. $O(n^2)$
- Key idea: If we divide list in two, there will be a majority of x in one of the subarrays. 

1. Recurse on division then at each level, figure out the majority element in the 2 subarrays. 
2. Return subarray with an element comprising n/2 of the subarray. By top level, with have majority element
3. log(n) levels, $O(n)$ comparisons at each level. Total $O(nlogn), \,T(n) = 2T(\frac{n}{2}) + O(n)$

## Randomized

### Collinear Lines on a Plane

- Idea: Points on a plane, want to find m, b s.t. the maximum number of points possible lies on that line.
- Brute force: Take every pair possible, compute line parameters in $O(n^2)$ . Sort and find the maximum with the same parameters, total $O(n^2log(n^2)) = O(n^2log(n))$
- Given a limitation on the input: Maximum number of collinear points is $\frac{n}{k}$ for constant k.
- Key idea: The limiting factor allows us to do something otherwise time expensive. In expectation it is a fast approach

1. Pick pair of points randomly and compute m and b. Check to see if every other point lies on this line. $O(n)$
2. If fewer than $\frac{n}{k}$ points lie on the line, repeat. 
3. The probability of selecting two points on the line ie. the number of iterations: $\frac{n/{k} \choose 2}{n \choose 2}$ or $\frac{n/k}{n} \times \frac{(n-1)/(k-1)}{n-1} = \frac{1}{k^2}\frac{n-k}{n-1} \leq \frac{1}{k^2} = O(k^{-2})$ 
4. This is a geometric RV with p = $\frac{1}{k^2}$, therefore in expectation = $k^2$ until our first success. Not a function of n so combined $O(k^2n) = O(n)$. Note that it is not guaranteed to terminate.

### Majority Element

* Idea: Array with $\frac{n}{2} +1$ elements with value x, can only make equality comparisons, want to find the majority element x.
* Brute Force: Run through array, comparing each value to every other until you find an element that is equal to a majority of the other elements. $O(n^2)$
* Key idea: Picking an element is a geometric variable with probability $\frac{n}{2}+1$. Randomized picking leads to a linear run time since the expectation of our geometric random variable is a constant.

1. Pick comparison element uniformly at random. Compare to every other element to determine if pick $\in X$. 
2. Repick until comparison works. Not guaranteed to terminate since a non-member of X could always be picked, but in expectation need to pick twice.

## Graphs

### Removing Vertex from CC

* Idea: Given an undirected graph with a single connected component, can you remove one vertex and maintain a single CC
* Key Idea: DFS can create a tree giving some structure to the relationships 

1. Run DFS and get a tree. 
2. Remove a leaf node and you can still get from any part of the tree to any other
3. Since the edges of the tree are a subset of the total edges, the graph is still connected
4. Note if we run DFS fully, runs in $O(n+m)$ but can reduce to $O(n)$ if we stop when we hit the first leaf.

### 2-SAT

* Idea: Given a 2-CNF, find an algorithm that can find an assignment that satisfies the formula
* Key Idea: If x and y are in the same SCC, then their negations will be a SCC together

1. Construct SCC DAG using FindSCC algorithm
2. TopoSort the DAG and reverse the ordering, so we are working from the leaves up
3. Keep cache of logical assignments. Working from the bottom assign True if negation not in cache, else assign False.

### Russian Boxes

* Idea: n boxes of different dimensions, boxes form a chain if they fit inside each other. Design an algorithm which takes as input a list of dimensions $w_i \times h_i$ and returns a longest possible chain of boxes
* Key Idea: The directed graph will be a DAG - there are no cycles of boxes fitting in other boxes

1. Apply TopoSort st any edge from $v_j \rightarrow v_i$, $j < i$
2. For each node $v_i$ , $\ell_{i}=1+\max _{\left(v_{j}, v_{i}\right) \in E} \ell_{j}$
3. Therefore starting from the smallest box, moving forward recoding # of boxes each node can fit inside. Each box takes the max of the nodes directed to it. 
4. Total is $\max _{i=1}^{n} \ell_{i}$

### Random Walks

* Idea: Given an undirected, connected graph G and two nodes s, t. Pick an edge at random starting from s, repeating at each node. Prove you will reach t.
* Key Idea: Connected graph / CC means there exists a path from s to t 
* Worst case scenario is a dense graph. Assign length n to shortest path from s to t. Chance of taking it is $\frac{1}{n^n}$. True for any vertex, so dividing time into epochs of length n steps, chance we never reach t is $\prod_1^\infty (1-\frac{1}{n^n}) = 0$

### Source Vertices

* Idea: A source vertex in a graph G = (V,E) is a vertex v such that all other vertices in G can be reached by a path from v. Say we have a directed, connected graph that has at least one source vertex. Find an algorithm to find a source vertex in $O(V + E)$
* Key Idea: If we run DFS, a source vertex must be last finished. 

### Bipartite Graphs

* Idea: Check whether a graph is bipartite - a graph whose vertices can be divided into two independent sets, U and V such that every edge (u,v) either connects a vertex from U to V or a vertex from V to U 
* Key Idea: BFS has layers of parent nodes and children nodes to check. DFS checks parents and children sequentially.

1. BFS: Start BFS from any node and color it RED. If its neighbor hasn’t been visited, color it the opposite color. If it has been visited, check if it is the opposite color of its parent. If not, return False. If we visit every node without False, then the graph is bipartite
2. DFS: Run DFS and traverse its tree, coloring each newly visited node the opposite color of its parent. If it has been visited before, check that its color is opposite of current node, else return False.

### Alien Alphabet

* Idea: Given list of n words sorted, determine the alphabet of length k that sorted the list.
* Key Idea: The letters are nodes on a graph, and the sorted order are directed edges

1. Create graph with k vertices for all possible letters in the alphabet. For adjacent words in the sorted list, compare the first letter that is different between them. The comparison of that letter determines an order, and an edge should be drawn on the graph from the left letter to the right. 
2. Will be left with a DAG. Simply TopoSort to get the alphabetical ordering
3. Runs in $O(k+ n)$ - comparing n-1 times over the words, drawing edges among k nodes. Using DFS / TopoSort same run time.

### Dijkstra’s Shortest Path

* Idea: single source shortest path, starting vertex s and **nonnegative** lengths for edges. Output the distance from s to every other vertex. While BFS computes shortest path in terms of edges, this considers shortest path in terms of weights (ie. BFS with expanded edges to correspond to weights)
* Key Idea: how do we choose the next vertex that we believe to be shortest?

1. Initialize X, set containing the vertices already visited. Initially assign length $\infty$ to all edges except starting vertex
2. For edge (v, w), $v \in X$ and $w \notin X$, choose an edge minimizing score $len(v) + l_{vw}$, ie. the length it took to get to v plus the weight between v and w. 
3. At each step, looks into X and compares the vertices to the next step of the graph, choosing the path the minimizes the score
4. Runs in $O(mn)$ time

## Dynamic Programming

### Longest Common Subsequence

* Idea: given two sequences made of a subset of letters, find the longest sequence of non-contiguous letters they have in common
* Brute Force: compute every possible common subsequence
* Key idea: start with the last letter of subsequence and consider cases that could define a previously longest sequence
* $O(mn)$ to fill in the table, $O(m+n)$ to recover LCS, total = $O(mn)$
* Optimal substructure: For strings X and Y, our sub-problems will be finding LCS’s of prefixes to X and Y

1. If $x_m = y_n$, then the last letter must be part of the LCS.
2. If $x_m \neq y_m$, at least one of these letters cannot appear in the LCS. Either $LCS(X, Y ) = LCS(X[1 : m − 1], Y ) $ or $LCS(X, Y ) = LCS(X, Y [1 : n − 1]) \rightarrow lenLCS(X, Y ) = max\{lenLCS(X[1 : m − 1], Y ), lenLCS(X, Y [1 : n − 1])\} $
3. Generate recurrence: $C[i, j]=\left\{\begin{array}{ll}{1+C[i-1, j-1],} & {\text { if } X[i]=Y[j]} \\ {\max (C[i-1, j], C[i, j-1]),} & {\text { otherwise }}\end{array}\right.$
4. Maintain an (n+1) by (m+1) table of entries. This is where the overlapping subproblems come in: we only need to compute each entry once, even though we may access it many times when filling out subsequent entries. This gives us the length of the LCS
5. Recover LCS: Work backwards through table, seeing if $X[i] = Y[j]$, $C[i,j] = C[i,j-1]$, or $C[i,j] = C[i-1,j]$ and decrementing relevant index. 

### Independent Set

* Idea: Say that we have an undirected graph $G = (V,E)$. We call a subset $S \subset V$ of vertices “independent”  if there are no edges between vertices in S. Let vertex $i$ have weight $w_i$, and denote $w(S)$ as the sum of weights of vertices in S. Graph must be a tree to be solvable in linear time. Looking for the indepedent set with the largest sum of weights.
* Optimal Substructure: Subtrees: either the root is in a MWIS or it is not. If it is not, use the optimal solution to the subtrees below it. If it is, then its direct children cannot be, so use the optimal solution from the small subproblems of its grandchildren.
* Pick vertex r and make it root. The vertex for subtree rooted at u is $T_u$ and $A(u)$ the weight of MWIS in $T_u$, $S_u$ the MWIS. 
* Either u is not in $S_u$, and $A(u)$ is the sum of $A(v)$ for all v children of u. 
* u is  in $S_u$, and $A(u)$ is the sum of $A(v)$ for all v children of u plus the weight of u.
* Call $\sum_v A(v) = B(u)$ for all children v in u
* Then $A(u)=\max \left\{\sum_{v \in \mathrm{CHLDREN}(u)} A(v), w_{u}+\sum_{v \in \mathrm{CHILDREN}(u)} B(v)\right\}$

### Zero Sum Subarray

* Idea: Given an array A of positive and negative integers, determine if there is a subarray with zero sum. 
* Key Idea: We can use the subsums from each previous entry in the array. Alternatively, we recognize that if two entries in the running sum are equal, the array between them must sum to 0.

1. DP approach: Loop through list, adding to running sum for each possible start position. If it equals zero with new addition, return True. If we precompute the running sums, we have two for loops and runs in $O(n^2)$
2. Hash approach: Compute running sum from index 0 to index j, done simply by adding element j to a variable of running sum. Check if value in hash table - if so return True. Else add running sum at index j to hash table. Runs in $O(n)$ since each step is constant amount of work, performed at worse for every element in the array.

### Longest Palindrome

* Idea: Given a string, compute the length of the longest palindrome that can be obtained by deleting some of its characters 
* Key Idea: Like LCS, start from first and last letters and work inwards. We have 2 cases, fit for recursion

1. Initiate cache dictionary to hold output of recursive steps
2. Define recursive function with starting and ending indices. Check if the indices under consideration are in the cache, and return that value. If length of string is 1, return 1.
3. If the first and last letter of the substring are equal, set value equal for those indices in the cache equal to 2 + recurse on str(1, n-1)
4. If the first and last are not the same, then take the max value returned by recursing on str(0, n-1) and str(1, n)
5. Run recursion and take the value returned. Runs in $O(n^2)$

### Optimal House Robber

* Idea: Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob if you cannot rob adjacent houses.
* Key Idea: At each index i, we can compare the max value already calculated for the i-2nd entry + value of i vs. the max value of the i-1 entry

1. Set initial values in caching array for 0th and 1st entries in list. 
2. Loop over 2 through n-1 and take max (A[i] + cache[i-2]) to (cache[i-1]).
3. Set as value for cache[i]
4. Return last value in cache array as max value possible.
5. Keeping track of a single index, runs in $O(n)$

### Letter Encoding

* Idea: Given numeric string S length n, develop O(n) algorithm to find how many letter strings can be created from the numbers.
* Key Idea: For single digit numbers could only be a single string. For alphabet up to two digits, can possibly create 1 or 2 letters at entry k using 1 or 2 digits

1. OS: running count of possible sums, sum will increase when two adjacent numbers form a number 10 - 26
2. Recurrence: $f(k)=sum\left\{\begin{array}{ll}{f(k-1)} & {\text { if } s[k] \in \{1,...,9\}} \\ {f(k-2)} & {\text { if }} s[k-1:k+1] \in [10,...,26]\end{array}\right.$
3. Run over list once, checking entry k and k-1 and adding to list of possibilities at entry k. Return last entry in list.

### Dice and Sums

* Idea: Find the probability that rolling k dice will result in a sum S
* Key Idea: The probability of getting S on ith roll is the probability that we have gotten a total of S-i from the previous rolls, summed across all possible i's
* Recurrence: $f(s, i)=\sum_{j=1}^{6} f(s-j, i-1) \cdot \frac{1}{6}$. There is a 1/6 probability of getting some number j on this ith roll.
* Loop over number of dice, loop over prior rolls, loop over possible values 1-6. If the prior_total + the roll is less than S, add to probability cache

### Rod Cutting

* Idea: Suppose we have a rod of length k, where k is a positive integer. We would like to cut the rod into integer- length segments such that we maximize the product of the resulting segments’ lengths. 
* Key Idea: Toggling the number of cuts and the length of the cuts. Therefore need to loop over the possible lengths of cuts up to k/2 and recurse on the remaining rod
* Recurrence: $f(k)=\max _{c \in\{2, k-1\}}(k, c \cdot f(k-c))$. k is rod length and c is number of cuts
* Notice that we do not need to consider cutting off a length of 1 since that will never yield the optimal product, and also do not need to try cuts any larger than ⌊k/2⌋ since those will already have been explored due to the symmetry of the cutting. 
* Base case is 1. For i in range 2 to k, the value to beat is no cuts. For cuts from 2 to i mod 2, the length remaining is i - c
* Compare no cutting to max prods[cut] x max prods[remaining] and take the max for either case. Fill in array for length of k the best product from the cuts.