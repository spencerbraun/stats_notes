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

### Integer Multiplication - Exam 1

* Idea: n positive integers of k digits each, want to compute their product
* Key Idea: Karatsuba computes products between two integers, divide and conquer karatsuba 

1. Partition the array into two sub arrays with n/2 integers each.
2. As the combine step, multiply the sub-products using Karatsuba
3. Run time for Karatsuba is $O(n^{log(3)})$. Product of two k-digit integers has ~2k digits. Product of n k-digit numbers has O(nk) digits. Recursion $T(n,k) = 2T(n/2,k) + O((nk)^{log(3)})$. By Master method get $O((nk)^{log(3)})$ total run time

### Batch Statistics - Exam 1

* Idea: input n large numbers and k distinct ranks for k < n. Output the $r_jth$ smallest of the n integers for j in {1,...,k}
* Key Idea: This is the lightbulb problem - making pivot comparisons between two groups.

1. Find median integer using select algorithm. Split integers relative to pivot. 
2. Split ranks based on comparison to rank of median.
3. Recurse separately on ranks and integers greater than pivot and those less than pivot.
4. Runs in $O(nlog(k))$. Log(k) depth to recursion since we halve the size of R at each iteration. Work at each level is O(n + k) = O(n). 

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
* Key Idea: If we run DFS, a source vertex must be last finished. Run DFS forest, find last finished. Check that it is a source by running BFS/DFS to ensure it reaches all other nodes

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

### Free Flights / Pretzels - Exam 2

* Idea: Input of a list of n airports, m flights from airport A to B, number of pretzels on each flight. Goal is to maximize the number of pretzels on any route.
* Key Idea: This is russian boxes mixed with DP. 

1. Create graph with airport vertices and directed edges as flights.
2. Run DFS to find any cycles - return infinity if cycles
3. Otherwise have a DAG. Run Toposort and reverse to get source flights.
4. Use DP to compute the max number of pretzels for a route ending at each airport - pretzels to A = $max_{B\text{ with flight to }A}$ (Pretzels to B) + (pretzels on B to A flight) or $P_i = i + max_{\text{over edges}} P_j$ for i current node and all incoming flights j.
5. Loop over i sorted airports. Set max neighbor to - infinity. For j in edges of i if pretzels bigger than max neighbor reassign. For each i store max pretzels in array.
6. DFS takes $O(n+m)$, DP outer loops n times, inner loop m times. Total $O(n+m)$

### Predicting Ties - Exam 2

* Idea: Given grid with weights, determine if two vertices could get to origin using same sum of weights. Can only move west and south, not north and east
* Key Idea: Since movement is restricted, we have optimal substructure. Use DP to compute shortest path to origin. Like BF, subset of shortest path is a shortest path.

1. Define optimal substructure. Let i,j define row and column in grid. For W list of west edges and S list of south edges, $d(i,j) = min[d(i-1,j) + W(i,j), d(i, j-1) + S(i,j-1)]$
2. Initialize hash table and DP distance table. Add base cases to the hash and DP table
3. For i, for j use DP table to calculate current shortest path. If that value is in the hash table, return the matching pair. Otherwise add the distance to the hash table. 
4. Return null if no match found.
5. Loops through for rows * columns in grid, here k x k. Total runtime $O(k^2)$

### Johnson’s Algorithm (APSP with Dijkstra)

* Idea: Given a graph with negative weights but no negative cycles, can use Dijkstra to find APSP
* Key Idea: Can transform negative weights to positive weights

1. Add a dummy vertex q, connected to every vertex with 0 cost edges, compute shortest path from q to other vertices
2. Use weights $w^{\prime}(u, v):=w(u, v)+h(u)-h(v)$ where h is the distance from q
3. For each vertex, run Dijkstra to get its SSSP
4. Run time is $nmlog(n)$ due to n calls to Dijkstra. 

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

## Greedy

### Activity Selection

* Idea: You can do one activity at a time and want to maximize the number of activies, Each activity a has a start time s and finish time f
* Order predetermined but activies to be performed can be chosen - need criterion for choosing
* Key Idea: Picking the activity with the smallest end time rules out the smallest number of additional activities.

1. At each step pick activity you can add with the smallest finish time
2. Runs in $O(n)$ if already sorted or $O(nlog(n))$
3. Suppose we have chosen $a_i$ and there is still an optimal solution $T^*$ that extends our choices. Consider next choice $a_k$ - if $a_k$ is not in $T^*$, imagine swapping $a_k / a_j$. Since $a_k$ has smallest ending time it ends before $a_j$, so it must not conflict with anything after $a_j$ in $T^*$. Then there must be an optimal schedule that also contains our choice $a_k$. 

### Job Scheduling

* Idea: n tasks, task i takes $t_i$ hours, for every hour until task i is done pay $c_i$ - includes time from earlier tasks. Find schedule that minimizes the cost
* All activities preselect but order can be chosen - need criterion for ordering
* Key Idea: This has nice optimal substructure where removing the next job leaves an optimal arrangement for the remaining. Cannot sort just by time or cost - need to take cost of delay / time it takes

1. Choose the job with biggest cost of delay / time it takes ratio
2. Suppose already chosen some jobs and optimal solution still alive. Say job B is the next best ratio job but not in the optimal solution. Switch A and B for A < B. Nothing else will change so cost of solution will not increase. Repeat until B is first - now we have the optimal schedule where B is first.

### Huffman Codes

* Idea: Prefix-free coding, no encoded string is a prefix of any other. Given frequency of letters in a population, whats the encoding that minimizes cost - the length of the encoding of that letter = frequency of letter times depth in tree. 
* Key Idea: Greedily build subtrees starting with infrequent letters to keep them at the bottom.

1. Create node for each letter / frequency, eg D: 16. 
2. While there are still nodes not in the tree, take the two nodes with the smallest keys 
3. Create a new node Z with key = X.key + Y.key
4. Assign X and Y as children to Z and now repeat with Z as a node to be considered instead of X and Y
5. Lemma1: Suppose x and y are two least frequent letters. Then there is an optimal tree where x and y are siblings
6. Lemma2: Can imagine that any Z is a leaf even though it has children

### Predicting Ties Revisited

* Idea: Now no restrictions on travel direction. Still want to predict ties to the origin
* Key Idea: We no longer have substructure if any direction can be considered. We need a single source shortest path from the origin

1. Run Dijkstra on the graph with origin as its source - the output is the shortest path from each vertex to the origin
2. Insert path lengths into hash table iteratively, if duplicate found there is a tie otherwise no ties.
3. Graph has $O(k^2)$ vertices and edges, so Dijkstra runs in $O(|E| + |V|log(|V|)) = O(k^2log(k))$. Hashing is constant time so duplicate check is $O(k^2)$. Overall run time is $O(k^2log(k))$

### Pareto Optimal

* Idea: Given a set of 2d points, Pareto optimal is a point (x, y) st for all (x’, y’) either x > x’ or y > y’. Find all Pareto optimal points.
* Key Idea: Sorting the array by x and using a greedy approach won’t keep optimal points ahead of us. Greedy often requires a certain ordering

1. Sort input points largest first by x, tiebreaking by y.
2. Initialize Pareto points with first point in the sorted list
3. Set Y to y coord of first point
4. For point in the remaining sorted array, if y > Y, then add that point to Pareto array and update Y to new y

### Cutting Ropes

* Idea: Given n ropes of different lengths and want to tie in to a single rope. Cost to connect two ropes is equal to the sum of their lengths. Want to connect all ropes with minimum cost.
* Key Idea: This is similar to job scheduling with cost = 1. Also like Huffman coding taking the min frequency.

1. Always combine smallest ropes available until there is one long rope.
2. Can use job scheduling with cost = 1 analogy and notice we only have a time parameter imposing a constant cost. Then obvious to sort by shortest time
3. Huffman: the frequency of letters is like a cost - higher frequency means we pay higher cost for more depth. Here the frequency is replaced by length of rope, but the cost analogy is the same. Use Huffman lemmas - if x and y are ropes with shortest length then there is an optimal tree where they are siblings. If we treat nodes at a given level as leaves can apply the lemma.

### Mice to Holes

* Idea: n mice and n holes along a line, stepping 1 unit left or right takes 1 minute. Assign mice to holes to minimize last mouse travel time.
* Key Idea: Try progressing from left to right

1. Sort the mice locations and hole locations. Have ith mouse go to ith hole. Max time will be max distance between all mice and their holes.
2. Lemma: $max(dist(a,c), dist(b,d)) \leq max(dist(a,d), dist(b,c))$ for mice $a < b$ and holes $c < d$
3. IH: By sending ith sorted mouse to the ith sorted hole, there is a minimal solution that extends the current solution
4. BC: If we have sent to mice to holes, the ideal solution has not been eliminated
5. IS: Suppose we have sent first k-1 mice to the first k-1 holes. Now imagine there is an optimal solution where kth mouse goes to $p_0th$ hole for k < $p_0$. Then each mouse is shifted aftert than until the $p_dth$ for some d mouse is sent ot the kth hole. We could then swap the kth and dth mice and result would not be worse than optimal.

## Min Cut / Max Flow

### Expense Settling

* Idea: k friends, fried i paid $c_i$. Want to develop algorithm to have everyone paid back where each person either pays or receives money but not both
* Key Idea: Money is like flow, where those that owe need to send and those owed receive. We have a bipartite graph

1. Calculate the per person average cost $\frac{\sum c_i}{k}$
2. Create a source s and sink t. Draw edges from s to those who owe and edges from owed to t with weights $c-c_i$ or $c_j -c$. 
3. Connect all nodes between the independent sets with weights infinity and run FF. 

### Project Selection with Prereqs

* Idea: set of k tasks $t_1,...,t_k$. Certain tasks such that $t_i$ is a prerequisite of $t_j$. Each task has reward $r_i$ which may be negative. Find optimal subset of tasks to maximize reward.
* Key Idea: A min cut will cut edges with the lowest weights and not cut edges with infinite weights. We want edges going from tasks to their prereqs due to the s-t cut.
* Suppose we have a task X with positive reward, and we have a task Y with a penalty (negative reward), I would recommend considering these three  cases to build intuition about what's going on: X is a prerequisite of  Y, Y is a prerequisite of X, or X and Y are unrelated. Case 1, our cut keeps x and discards Y, which works because we only have an edge from t to s side of cut. Case 2, XY on S side if reward for x outweighs penalty for Y or on t side if not - the cut chooses the s-x, y-t edge that is lighter. If unrelated, our cut keeps x and excludes Y (holding all else equal).

1. Draw edges from prereqs to tasks with edge weights infinity
2. Add source and sink, s and t
3. Draw edge from s to vertex with weight r if r > 0
4. Draw edge from vertex to t with weight r if r < 0
5. Draw min s-t cut and take tasks on the s side of the cut.

### Tiling Partial Checkerboard

* Idea: n x n checkerboard with some squares deleted. Each domino can cover two squares. Determine whether one can tile the board completely.
* Key Idea: Can we separate into bipartite groups? Perhaps black and white squares with edges for adjacency. Now just check for a perfect matching

1. Make G bipartite with black and white ind. sets with adjacency edges. $O(n^2)$ edges and vertices
2. Compute maximum matching. If perfect we can cover else cannot
3. FF runs in $O(|f|E) = O(n^4)$, which dominates the run time

### Ice Cream Matching

* Idea: People and ice cream tubs. Each person can only eat c(x) scoops. Each tub of ice cream only has c(y) scoops in it. Each pair can only be matched c(x,y) times - ie. student x only wants 3 scoops of flavor y. Assign as many matches as possible
* Key Idea: Desired scoops are the edges between sets. c(x) weights from s to people, c(y) weights from ice cream to t.

1. Design graph as above and run F-F to find max bipartite matching
2. Flows correspond to assignments and max flows correspond to max assignments 

## Section 8 Problems

### Investing

* Idea: Want to buy low and sell high, array A of integers of future prices and can make one buy followed by one sell. What is the max profit? Design one divide and conquer and one O(n) algorithm.
* D&C: Need minimum price before the maximum price. Divide down to single prices, base case is profit = 0. Recurse finding max profit in L and max profit in R. Max profit across is max R  - min L. Total max profit in max(Max L, Max R, Max Across) and returns max profit. $T(n) = 2T(n/2) + O(n)$
* O(n) algorithm: Initialize min value and max profit variables. Loop through array, for each element calculate profit and set equal to max profit if bigger, calculate min value and set it to min value if smaller. Constant work, one loop through prices. Since min value only reset after profit calculations, always have a buy followed by a sell.

### Quicksand

* Idea: M x N grid, some vertices are quicksand and adjacent vertices are also unsafe. At each time can travel to any adjacent location. Design algorithm that returns a shortest safe path from one side on left to any location on right side of grid.
* Key Idea: Construct undirected unweighted grid with adjacency edges. Remove quicksand vertices, neighbors, and associated edges - 5 vertices per pit and 4 + 3*4 = 16 edges per pit. Treated as constant time per pit. Add source and sink with source connected to leftmost column and sink connected to rightmost. Run BFS from S.
* Graph construction $O(MN)$ time since order of vertices plus edges. Pit removal O(V) = O(MN) since constant work. O(M) to add source and sink since edges just to number of rows. Running BFS takes O(MN)

### Graph Coloring

* Idea: Using greedy algorithms to color a graph with adjacent edges with different colors - use d+1 colors where d is max degree of a vertex.
* Algorithm: Pick vertex with highest degree and color. For each iteration check if neighbors colored and color by trying remaining colors in same order. Loops through every node and WC looks at every other node and edge so $O(V^2 + E)$

### Covering a Number Line

* Idea: n is initial position of person on a number line. l is probability of person going left. Find the probability of reaching all points on the number line after n moves. 
* Solution: Think of knight moves. Probability of reaching i after j moves is $A[i][j]$. $A[i][j] = (l)A[i+1][j-1] + (1-l)A[i-1][j-1]$. Initialize with $A[i][0]$ for i=n, 0 otherwise.
* So j goes from 0 to n, since number of possible moves. Can only reach up to 2n after n moves, so runs in $O(n^2)$. 

### Minimum Spanning Tree with Constraint

* Idea: weighted undirected graph G with edge weights in W = {1,2,3}, design algorithm to find an MST. 
* Solution: Run Kruskal with RadixSort on the edges. Since weights are capped RadixSort is possible. Runs in $O(m)$