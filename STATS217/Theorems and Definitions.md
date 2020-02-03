# Theorems and Definitions

# Durrett

## Multistep Transition Probabilities

* Absorbing state: $p(x,x) = 1$
* **D Theorem 1.1.** The m step transition probability $P\left(X_{n+m}=j | X_{n}=i\right)$ is the mth power of the transition matrix p.
* Chapman-Kolmogorov Equation: $p^{m+n}(i, j)=\sum_{k} p^{m}(i, k) p^{n}(k, j)$ 
  * Intuitively, to go from i to j in m + n steps, we have to go from i to some state k in m steps and then from k to j in n steps. The Markov property implies that the two parts of our journey are independent.

## Classification of States

* First time to return to y: $T_{y}=\min \left\{n \geq 1: X_{n}=y\right\}$
  * Does not count time zero
* Probability of process returning to y starting at y: $\rho_{y y}=P_{y}\left(T_{y}<\infty\right)$
* Stopping time: $\{T=n\}$ 
  * The occurrence (or nonoccurrence) of the event “we stop at time n,” can be determined by looking at the values of the process up to that time: $X_{0}, \ldots, X_{n}$
* **D Theorem 1.2** (Strong Markov Property). Suppose T is a stopping time. Given that $T=n$ and  $X_t = y$, any other information about $X_{0} \ldots X_{T}$ is irrelevant for predicting the future, and $X_{T+k}, k \geq 0$ behaves like the Markov chain with initial state y.
* Transient: $\rho_{y y}<1$
  * The probability of returning k times is $\rho_{y y}^{k} \rightarrow 0 \text { as } k \rightarrow \infty$, eventually the Markov chain does not find its way back to y.
* Recurrent: $\rho_{y y}=1$ 
  * The probability of returning k times $\rho_{yy}^k =1$ so the chain returns to y infinitely many times.
  * An absorbing state is the strongest version of recurrence $p(y,y) =1$
* D Lemma 1.3. Suppose $P_{x}\left(T_{y} \leq k\right) \geq \alpha>0$ for all x in the state space S. Then $P_{x}\left(T_{y}>n k\right) \leq(1-\alpha)^{n}$ 

* D Definition 1.1. We say that x communicates with y and write $x \rightarrow y$ if there is a positive probability of reaching y starting from x, that is, the probability $\rho_{x y}=P_{x}\left(T_{y}<\infty\right)>0$
  * In english, we can travel to y starting from x with positive probability
* D Lemma 1.4. If$ x \rightarrow y$ and $y \rightarrow z$, then $x \rightarrow z$. Communication is transitive. 

* **D Theorem 1.5**. If $\rho_{x y}>0$ but $\rho_{yx} < 1$ then x is transient
  * This allows us to identify all the transient states when the state space is finite.
* Lemma 1.6. If x is recurrent and $\rho_{x y}>0$ then $\rho_{y x}=1$
  * Directly follows from T1.5. 
* Closed: A set A is closed if it is impossible to get out.  $p(i, j)=0$ for $i \in A, j \notin A$
* Irreducible: set B irreducible if whenever $i, j \in B$ i and j communicate. 
  * In english, a set of states that all communicate with each other. 
  * The whole MC is irreducible if all states communicate with each other.

* **D Theorem 1.7**: If C is a <u>finite closed and irreducible</u> set, then all states in C are recurrent. 
* **D Theorem 1.8** (Decomposition Theorem): If the state space S is **finite**, then S can be written as a disjoint union $T \cup R_{1} \cup \cdots \cup R_{k}$ where T is a set of transient states and the $R_i$, $1 \leq i \leq k$, are closed irreducible sets of recurrent states.
  * For finite states, we can create groups based on which states communicate with each other.
* Lemma 1.9. If x is recurrent and $x \rightarrow y$, then y is recurrent.
* Lemma 1.10. In a <u>finite closed</u> set there has to be at least one recurrent state.

* Lemma 1.11 / 1.12. $E_{x} N(y)=\rho_{x y} /\left(1-\rho_{y y}\right) =\sum_{n=1}^{\infty} p^{n}(x, y)$. 
  * Expected number of visits to y starting from x
  * Recurrence defined by $E_{y} N(y)=\infty$, expected # of time periods that process is in state y is infinite.
* **D Theorem 1.13**. y is recurrent if and only if $\sum_{n=1}^{\infty} p^{n}(y, y)=E_{y} N(y)=\infty$
  * Necessary and sufficient condition for recurrence. 
  * In english, y is recurrent if the expected number of visits to y starting at y is infinite. Alternatively, y is recurrent if the infinite sum of transition probabilities from y to y is infinite.

## Stationary Distributions

* Define: $q(i)=P\left(X_{0}=i\right)$ 
* Stationary Distribution: If $\pi p = \pi$, then $\pi$ is called a stationary distribution. If the distribution at time 0 is the same as the distribution at time 1, then by the Markov property it will be the distribution at all times
  * That after all the transfers are made, the amount of sand that ends up at each site is the same as the amount that starts there. Not necessarily from x to y and y to x - transfers around the whole system.

* Definition 1.2. A transition matrix p is said to be doubly stochastic if its COLUMNS sum to 1
* **D Theorem 1.14**. If p is a doubly stochastic transition probability for a Markov chain with N states, then the uniform distribution, $\pi(x)=1 / N$ for all x, is a stationary distribution.
* Detailed balance condition: $\pi(x) p(x, y)=\pi(y) p(y, x)$
  * A stronger condition than the stationary distribution. In this case the detailed balance condition says that the amount of sand going from x to y in one step is exactly balanced by the amount going back from y to x.

## Reversibility

* Theorem 1.15: Fix n and let $Y_{m}=X_{n-m} \text { for } 0 \leq m \leq n$. Then $Y_m$ is a Markov chain with transition probability $\hat{p}(i, j)=P\left(Y_{m+1}=j | Y_{m}=i\right)=\frac{\pi(j) p(j, i)}{\pi(i)}$. 
  * In english, if we watch the process $X_{m}, 0 \leq m \leq n$ backwards, then it is a MC. For transition probability $p(i,j)$ with stationary distribution $\pi(i)$ and Let Xn be a realization of the Markov chain starting from the stationary distribution, $P\left(X_{0}=i\right)=\pi(i)$

## Limit Behavior

* Period: The period of a state is the largest number that will divide all the n $\geq$ 1 for which $p^{n}(x, x)>0$
* Lemma 1.17. If $\rho_{x y}>0 \text { and } \rho_{y x}>0$, then x and y have the same period - periodicity is a class property. 
  * If you show aperiodicity for some states in an irreducible chain, then the whole chain is aperiodic.
* Lemma 1.18. If $p(x, x)>0$ then x has period 1.
  * Very useful in showing aperiodicity as a sufficient condition. 
* Define 
  * I: p is irreducible, 
  * A: aperiodic (all states have period 1), 
  * R: all states recurrent, 
  * S: stationary distribution $\pi$ exists
* **D Theorem 1.19 Convergence Theorem**: $p^{n}(x, y) \rightarrow \pi(y)$ as $n \rightarrow \infty$ for I, A, S
* **Theorem 1.20 (Asymptotic Frequency).** Suppose I and R. R. If $N_n(y)$ be the number of visits to y up to time n, then $\frac{N_{n}(y)}{n} \rightarrow \frac{1}{E_{y} T_{y}}$
  * The average probability of visiting y converges to the 1 over the expected min time to return to y starting from y.
  * If state space is infinite, we can have $E_{y} T_{y}=\infty$ and the limit is 0
* **Theorem 1.21**. If I and S hold, then $\pi(y)=1 / E_{y} T_{y}$ and hence the stationary distribution is unique.
* **Theorem 1.22**. Suppose I, S, and $\sum_{x}|f(x)| \pi(x)<\infty$ then $\frac{1}{n} \sum_{m=1}^{n} f\left(X_{m}\right) \rightarrow \sum_{x} f(x) \pi(x)$
  * Just extending 1.20
* **Theorem 1.23**. Suppose I, S. $\frac{1}{n} \sum_{m=1}^{n} p^{m}(x, y) \rightarrow \pi(y)$
  * The average of the first n values of $p^{m}(x, y)$ converges even in the periodic case
* Ergodic: State i is positive recurrent if i is recurrent and, starting in i, the expected return time to i is finite. Positive recurrent, aperiodic states are called ergodic.
* Limiting Probabilities: An irreducible, ergodic MC has limit independent of i $\pi_{j}=\lim _{n \rightarrow \infty} P_{i j}^{n}$, where $\pi_j$ is the solution to the system  $\pi_{j}=\sum_{i=0}^{\infty} \pi_{i} P_{i j}, \quad j \geq 0 \text{ s.t. }\sum_{j=0}^{\infty} \pi_{j}=1$. Note $\pi_j$ also the long run proportion of time that the process will be in state j

## Exit Times

* **Theorem 1.29**. Let $V_{A}=\inf \left\{n \geq 0: X_{n} \in A\right\}$. Suppose $C=s-A$ is finite, and that $P_{x}\left(V_{A}<\infty\right)>0$ for any x in C. If g(a) = 0 for all a in A, and for x in C  we have $g(x)=1+\sum_{y} p(x, y) g(y)$, then $g(x)=E_{x}\left(V_{A}\right)$.
  * 

# Pinsky

## Limiting Distributions

* **Regular**: On a finite number of states labeled 0,1,...,N, P has the property that when raised to some power k, the matrix $P^k$ has all of its elements strictly positive.
  * Not all Markov chains are regular. The identify matrix remains in its initial state - it has a limiting distribution but it clearly depends on its initial state

* Limiting Probability Distribution: $\pi=\left(\pi_{0}, \pi_{1}, \ldots, \pi_{N}\right), \text { where } \pi_{j}>0 \text { for } j=0,1, \ldots, N \text { and } \Sigma_{j} \pi_{j}=1$
  * Primary interpretation: $\pi_{j}=\lim _{n \rightarrow \infty} P_{i j}^{(n)}=\lim _{n \rightarrow \infty} \operatorname{Pr}\left\{X_{n}=j | X_{0}=i\right\}$. In words, after the process has been in operation for a long duration, the probability of finding the process in state j is $\pi_j$, irrespective of the starting state.
  * Secondary interpretation: $\pi_j$ gives the long run mean fraction of time that the process {Xn} is in state j.
* For a regular transition probability matrix, $\lim _{n \rightarrow \infty} P_{i j}^{(n)}=\pi_{j}>0 \quad \text { for } j=0,1, \ldots, N$
  * In the long run the probability of finding the Markov chain in state j is approximately $\pi_j$ no matter in which state the chain began at time 0.
* Sufficient conditions for regularity:
  * For every pair of states i,j there is a path $k_1,...,k_r$ for which $P_{i k_{1}} P_{k_{1} k_{2}} \cdots P_{k_{j} j}>0$ - ie. you can reach any state through transitions of others
  * There is at least one state i for which $P_{ii} > 0$ - diagonals have at least one positive entry
* Theorem 4.1: Let P be a regular transition probability matrix on the states 0,1,...,N. Then the limiting distribution $\pi=\left(\pi_{0}, \pi_{1}, \ldots, \pi_{N}\right)$ is the unique nonnegative solution of the equations $\pi_{j}=\sum_{k=0}^{N} \pi_{k} P_{k j}, \quad j=0,1, \ldots, N$ and $\sum_{k=0}^{N} \pi_{k}=1$
  * In english limiting distribution sets up a system of equations to be solved

## Classification of States

* Accessible: j is said to be accessible from state i if $ P_{i j}^{(n)}>0$ for some integer $n \geq 0$; i.e., state j is accessible from state i if there is positive probability that state j can be reached starting from state i in some finite number of transitions.
* Communicate: When two state i and j are each accessible to each other. 
* Equivalence classes: The states in an equivalence class are those that communicate with each other. 
  * It may be possible starting in one class to enter some other class with positive probability; if so, however, it is clearly not possible to return to the initial class, or else the two classes would together form a single class.
* Period: We define the period of state i, written d(i), to be the greatest common divisor (g.c.d.) of all integers $n \geq 1$ for which $P_{i i}^{(n)}>0$ 
  * If $P_{i i}>0$ for some single state i, then that state now has period 1 and is called aperiodic
  * Can look at ii entry in P at increasing powers of P to see period
  * Property 1: If i communicates with j, the d(i) = d(j) - period is constant in each class 
  * Property 2: State i with period d(i), then an integer N exists such that $P_{i i}^{(n d(i))}>0$ - a return to state i can occur at all large multiples of the period d(i)
  * Property 3: If $P_{j i}^{(m)}>0, \text { then } P_{j i}^{(m+n d(i))}>0$ for all n sufficiently large.
* Return at nth transition: $P_{i i}^{(n)}=\sum_{k=0}^{n} f_{i i}^{(k)} P_{i i}^{(n-k)}, \quad n \geq 1$
  * When process starts at i, the probability it returns to state i at some time is $f_{i i}=\sum_{n=0}^{\infty} f_{i i}^{(n)}=\lim _{N \rightarrow \infty} \sum_{n=0}^{N} f_{i i}^{(n)}$
* Recurrence: $f_{i i}=1$, otherwise transient. 
  * A state i is recurrent if and only if, after the process starts from state i, the probability of its returning to state i after some finite length of time is one.
* Theorem 4.2: A state i is recurrent if and only if $\sum_{n=1}^{\infty} P_{i i}^{(n)}=\infty$
  * Conversely, suppose $\sum_{n=1}^{\infty} P_{i i}^{(n)}<\infty$. Then, M is a random variable whose mean is finite, and thus, M must be finite. That is, starting from state i, the process returns to state i only a finite number of times.

## Limit Theorem of MCs

* For recurrent state i, $f_{ii}^{(n)}$ is the probability distribution of the first return time, $R_{i}=\min \left\{n \geq 1 ; X_{n}=i\right\}$. $f_{ii}=1$ and $R_i$ is also finite.

* Mean duration between visits to state i is $m_{i}=E\left[R_{i} | X_{0}=i\right]=\sum_{n=1}^{\infty} n f_{i i}^{(n)}$

  * After starting in i, then, on the average, the process is in state i once every $m_i$ units of time

* Theorem 4.3 (Basic Limit Theorem): Consider a **recurrent irreducible aperiodic** Markov chain. Let $P_{ii}^{(n)}$ be the probability of entering state i at the nth transition given $X_0=i$ (initial state is i). Let $f_{ii}^{(n)}$ be the probability of first returning to state i at the nth transition for n = 0,1,... where $f_{ii}^{(0)}=0$ then $\lim _{n \rightarrow \infty} P_{i i}^{(n)}=\frac{1}{\sum_{n=0}^{\infty} n f_{i i}^{(n)}}=\frac{1}{m_{i}}$

  * Under the same conditions as in (a), $\lim _{n \rightarrow \infty} P_{j i}^{(n)}=\lim _{n \rightarrow \infty} P_{i i}^{(n)}$ for all states j. 
  * The limit theorem applies verbatim to any aperiodic recurrent class.

* Positive Recurrent: $\lim _{n \rightarrow \infty} P_{i i}^{(n)}>0$ for one i in an aperiodic recurrent class, then $\pi_j > 0$ for all j in the class of i. The class is then positive recurrent or strongly ergodic. 

  * State i is pos. recurrent if $m_{i}=E\left[R_{i} | X_{0}=i\right]<\infty$

* Null Recurrent: If each $\pi_i = 0$ and class is recurrent, then null recurrent or weakly ergodic.

  * State i is null recurrent if $m_{i}=\infty$
  * Follows from $\lim _{n \rightarrow \infty} P_{i i}^{(n)}=\pi_{i}=1 / m_{i}$

* Theorem 4.4: In a **positive recurrent aperiodic class** with states j = 0,1,2,..., $\lim _{n \rightarrow \infty} P_{i j}^{(n)}=\pi_{j}=\sum_{i=0}^{\infty} \pi_{i} P_{i j}, \quad \sum_{i=0}^{\infty} \pi_{i}=1$ and the $\pi’s$ are are uniquely determined by the set of equations $ \pi_{i} \geq 0, \sum_{i=0}^{\infty} \pi_{i}=1$ and $\pi_{j}=\sum_{i=0}^{\infty} \pi_{i} P_{i j} \quad \text { for } j=0,1, \ldots$

  * These $\pi_i’s$ are called a stationary probability distribution of the MC. The term “stationary” derives from the property that a Markov chain started according to a stationary distribution will follow this distribution at all points of time.
  * When initial state is selected according to the stationary distribution, then $\operatorname{Pr}\left\{X_{n}=i, X_{n+1}=j\right\}=\pi_{i} P_{i j}$

* Periodic Case: If i is a member of a recurrent periodic irreducible Markov chain with period d, one can show that $P_{i i}^{m}=0$ if m is not a multiple of d and that for multiple of d n $\lim _{n \rightarrow \infty} P_{i i}^{n d}=\frac{d}{m_{i}}$

* A **unique stationary distribution** $\pi=\left(\pi_{0}, \pi_{1}, \ldots .\right)$ exists for a **positive recurrent periodic irreducible** Markov chain, and the mean fraction of time in state i converges to $\pi_i$ as the number of stages n grows to infinity. The convergence of does not require the chain to start in state i.

  

