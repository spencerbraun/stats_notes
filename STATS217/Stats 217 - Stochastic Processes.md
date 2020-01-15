---
title: STATS 217
date: 01/06/2019
---



[TOC]

# Stats 217 - Stochastic Processes

## Motivating Examples

* Random Walks 
  * Drunkard on the street, modeling their walk on the real number line (integer lattice $\Z$)
  * Starting at 0 (initial condition), steps left and right with equal probability, each step is independent of all other steps
  * Questions to answer: 
    * What is the probability drunkard reaches home eventually, say position -10. 
    * Expected return time to 0?
    * What is the probability he reaches home before falling down manhole, say at position 5
    * Bring in higher dimensions - d=2, d=3 same questions. There are significant qualitative differences to these questions in different dimensions. In dimensions 1, 2 will return to 0 with probability 1, but not in 3 or higher.
  * Other graphs - $T_d$ - infinite d-regular tree. After each step only a unique edge to return to prior position - tend to float away towards infinity for trees with higher degree nodes.
  * Can approximate large finite graphs like social networks with infinite graphs 
* Arrivals Process
  * People arrive at a phone booth at a rate of 10 people / hr. Spend ~ 5 minutes in the booth. 
  * Can introduce independence between people to create a model. Can rely on a distribution for minutes in the booth instead of average. Then can answer questions like how long on average is the queue.
* Branching Processes
  * Galton-Watson Process - used to model human Y-chromosome DNA haplogroups - transferred from fathers to sons. Alternatively, how long does a surname last if passed from father to progeny. 
  * Randomness introduced by mutations. Haplogroup defined by a set of alleles, after a certain number of mutations no longer in the same group.
  * Each man has $Pois(\mu)$ sons, independent of others. What is the probability of name extinction for varying values of $\mu$? 

## Probability Review (Chapters 1 and 2)

### Probability Spaces

* Sample Space: $\Omega$ - set of outcomes 
* $\sigma$-algebra of events: $F, \; A \subset \Omega$
  * Algebra - some set along with some operations that can be applied
  * $\empty = \{\} \in F$
  * If $A  \in F,\implies A^C = \{\omega \in \Omega: \omega \notin A\} \in F$ - if event A in F so is its complement
  * $A_1, A_2,... \in F \implies \cup^\infty_{n=1} A_n \in F$ - if sequence of events in F then so is their union
  * Note: $(A^C \cup B^C)^C = A \cap B$ - then intersections are also in the sigma-algebra
* Probability Measure: $P: F \rightarrow [0,1]$. Input an event and it outputs a probability
  * $P(\empty) = 0$
  * Additivity: $A_1, A_2, A_3$ are pairwise disjoint ($A_i \cap A_j = \empty \; \forall i \neq j$), then $P(\cup_{n=1}^\infty A_n) = \sum_{n=1}^\infty P(A_n)$. In general without disjoint assumption: $P(\cup_{n=1}^\infty A_n) \leq \sum_{n=1}^\infty P(A_n)$ (union bound). This is intuitively since we are counting the intersection multiple times.
  * Consequences: $P(A^C) = 1 - P(A)$, $P(\Omega) =1, \; P(A) \leq 1 \; \forall A$

### Random Variables

* Function from $\Omega \rightarrow \R$
* CDF $F_X(u) = P(X \leq u)$. $X \leq u$ is the set $\{\omega \in \Omega: X(\omega)\leq u\} \in F$

* Example: $\Omega = \{(a_i)_{i=1}^\infty: a_i \in \{-1, +1 \} \}$ - space for random walk. Then $X_i = X_i(\omega) = a_i$. Define sum$S_n = \sum_{j=1}^nX_j(\omega)$. Hitting time $T_{10} = inf\{n \geq 1: S_n = 5\}$

##### Discrete RVs

* X is discrete: has a countable set of outcomes - can be labeled by integers $\{x_i\}_{i=1}^\infty$
* PMF: $P(X=x) = p_X(x)$, and $\sum_x p_X(x) = 1$ for $p_X(x) \geq 0$

##### Continuous RVs

* By defn, X continuous has a PDF: $f_X(x)$ st $\forall a < b \in \R,\; P(a \leq X \leq b) = \int_a^b f_X(x)dx$
* CDF $F_X(x) = P(X \leq x) = \begin{cases} \sum_{y \leq x} p_X(y) & \text{discrete} \\ \int_{-\infty}^\infty f_X(y) dy & \text{continuous}\end{cases}$

##### Joint Distributions

* $X_1,...,X_n$ RVs, Joint CDF $F_{X_1,...,X_n}(x_1,...,x_n) := P(X_1 \leq x_1,..., X_n \leq x_n)$
* Discrete: $P_{X_1,...,X_n}(x_1,...,x_n) = P(X_1=x_1,...,X_n=x_n)$
* Continuous: $P(X_1 \in[a_1,b_1],...,X_n \in [a_n,b_n]) = \int_{a_1}^{b_1}...\int_{a_n}^{b_n}f_{X_1,...,X_n}(x_1,...,x_n)dx_n...dx_1$
* Events: $A_1,...,A_n \in F$ are (mutually) independent if $P(A_1 \cap ... \cap A_n) = \prod_{i=1}^nP(A_i)$
  * Pairwise independent if $P(A_i \cap A_j) = P(A_i)P(A_j) \forall i,j$. But almost always refer to mutual independence
* Variable independence if $F_{X_1,...,X_n}(x_1,...,x_n) = \prod_{i=1}^nF_{X_i}(x_i), \; \forall x_1,...,x_n$. Alternatively, $P(X_1 \in S_1,...,X_N \in S_N) = \prod_{i=1}^nP(X_i \in S_i), \; \forall Si=[a_i, b_i]$ for sets S

### Conditional Probability

* Events A,B with $P(B) \neq 0$, $P(A|B) = \frac{P(A \cap B)}{P(B)}$

##### Discrete

* Conditional PMF: $p_{X | Y}(x | y)=\frac{p_{X Y}(x, y)}{p_{Y}(y)}$ . Law of total probability: $\operatorname{Pr}\{X=x\}=\sum_{y=0}^{\infty} p(X=x | Y=y)P(Y=y)=\sum_{y=0}^{\infty} p_{X | Y}(x | y) p_{Y}(y)$
* Conditional expectation with specified value (Y=y is an event): $E(X | Y = y) = \sum_x xp_{X|Y}(x|y)$
* Conditioning expectation without an event. Denote $\phi(y) = E(X|Y=y)$, then $\phi(Y)$ is a RV $E(X|Y)$. 
  * For example, $X \sim Unif(\{1,...,6\})$ outcome of die roll. $Y = \begin{cases}1 &if \; X \; even \\ 0 & if \; X \; odd\end{cases} = 1(X \;even)$. Then $E(X|Y) = \begin{cases}4 & \text{on Y =1} \\3 & \text{on Y = 0}\end{cases} = 4 \times1(Y=1) + 3\times1(Y=0) = 4Y + 3(1-Y) = 3 + Y$. Clearly this is a function of Y 
  * $E(X) = E[E(X|Y)] = \sum_y p_Y(y)E(X|Y=y) = \sum_y P(Y=y+\sum_x xP(X=x|y=y) = \sum_x x\sum_yP(Y=y)P(X=x|Y=y)$. Summed over all y, $\sum_yP(Y=y)P(X=x|Y=y) = P(X=x)$, so the expression reduces to $E(X)$
* Conditional expectation: $E[g(X) | Y=y]=\sum_{x} g(x) p_{X | Y}(x | y)$. Law of total probability: $E[g(X)]=\sum_{y} E[g(X) | Y=y] p_{Y}(y) = E\{E[g(X) | Y]\}$. In its final form, this is a function of the RV Y.

##### Continuous
* $f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$

##### Random Sums

* $X=\xi_{1}+\cdots+\xi_{N}$ where N is discrete RV with pmf $p_{N}(n)=\operatorname{Pr}(N=n)$, N > 0.
*  Mixed continuous X / discrete N
  * $F_{X | N}(x | n)=\frac{\operatorname{Pr}\{X \leq x \text { and } N=n\}}{\operatorname{Pr}\{N=n\}}$
  * Law of total probability: $f_{X}(x)=\sum_{n=0}^{\infty} f_{X | N}(x | n) p_{N}(n)$
* Moments assuming ${E\left[\xi_{k}\right]=\mu,}  \;{\operatorname{Var}\left[\xi_{k}\right]=\sigma^{2}},\;
  {E[N]=v,} \;{\operatorname{Var}[N]=\tau^{2}}
  $
  * $E[X]=E[\xi_k]E[N]=\mu v$
  * $\operatorname{Var}[X]=E[N]Var[\xi_k] + E[\xi_k]^2Var[N]=v \sigma^{2}+\mu^{2} \tau^{2}$
* Distribution - n-fold convolution of the density f(z), denoted $f^{(n)}(z)$
  * For $\xi_{1}, \xi_{2}, \ldots$ continuous RVs with PDFs f(z)
  * $f^{(1)}(z)=f(z)$
  * $f^{(n)}(z)=\int f^{(n-1)}(z-u) f(u) \mathrm{d} u \quad \text { for } n>1$
  * X is continuous and has marginal density $f_{X}(x)=\sum_{n=1}^{\infty} f^{(n)}(x) p_{N}(n)$

##### Martingales

* Definition

  * A stochastic process is a martingale if for n = 0, 1, ...

  1. $E\left[\left|X_{n}\right|\right]<\infty$
  2. $E\left[X_{n+1} | X_{0}, \ldots, X_{n}\right]=X_{n}$

* By taking expectations of second condition, see martingale has constant mean: $E\left[X_{0}\right]=E\left[X_{k}\right]=E\left[X_{n}\right], \; 0 \leq k \leq n$

* Markov Inequality: for non negative RV X and positive constant $\lambda$, $\lambda \operatorname{Pr}\{X \geq \lambda\} \leq E[X]$

* Maximal Inequality Theorem: 

  * For a nonnegative martingale, $\operatorname{Pr}\left\{\max _{0 \leq n \leq m} X_{n} \geq \lambda\right\} \leq \frac{E\left[X_{0}\right]}{\lambda}$ for $0 \leq n \leq m$
  * $\operatorname{Pr}\left\{\max _{n \geq 0} X_{n}>\lambda\right\} \leq \frac{E\left[X_{0}\right]}{\lambda}$ for all n
  * The maximal inequality limits the probability of observing a large value anywhere in the time interval 0,...,m, ie. inequality limits the probability of observing a large value at any time in the infinite future of the martingale

### Class Examples

1. Conditioning 
   * Q: Alice and Bob take turns flipping a coin. First to flip heads wins - first mover has an advantage. What is the probability that Alice wins if she goes first? 
   * Let T = # of rounds until a heads is flipped (by A or B). Event $\{T=t\}$ splits into 3 events: 
     * 2(t-1) tails (before t^th round) followed by $\begin{cases}HH & A \; wins\\HT & A \; wins\\TH & B \; wins\end{cases}$, each has prob $(1/2)^{2t}$ - power of number of flips. $P(A \;wins | T=t) = \frac{P(A \;wins,T=t)}{P(T=t)} = \frac{2(0.5)^{wt}}{3(0.5)^{2t}} = \frac{2}{3}$
     * $P(A \; wins) = \sum_{t=1}^\infty P(A \;wins |T=t)P(T=t) = \frac{2}{3}\sum_{t=1}^\infty P(T=t) = \frac{2}{3}$
  
2. Branching
  * Q: Family with N distributed $Pois(\lambda)$. Each child has blue eyes with prob 1/4, brown with prob 3/4, independent of other children and N. Let Z = # of blue eyed children. What is E(Z)?
  * Let $X_i = 1(\text{ith child has blue eyes})$. Then $Z = \sum_{i=1}^N X_i$ - we have a sum with a random upper limit. But we can condition on the RV N: $E(Z|N) = E(\sum_{i=1}^N X_i| N) = \sum_{i=1}^NE(X_i|N)$. Note the summation becomes deterministic conditioned on N, so we can pull out of the expectation. X and N are independent, so $=\sum_{i=1}^NE(X_i) = N * \frac{1}{4} = N/4$ 
  * $E(Z) = EE(Z|N) = \frac{1}{4}EN = \frac{\lambda}{4}$

## Chapter 3: Markov Chains

* A stochastic process is a family of RVs $(X_t)_{t \in T}$ indexed by a set T (say time) characterized by 1) indexed set T, 2) state space S, the set of possible outcomes $X_t$, 3) Joint distributions of finite dimensional marginals, $X_{t_1},...,X_{t_n},\; \forall t_1,...,t_n \in T$
* The joint distribution: need to specify $P(X_{t_1} \in A_1,...,X_{t_n} \in A_n)$ for all t in T and all A in S. But generally enough to know what the pmf is. For $X_t$ discrete (S finite) enough to specify the joint PMFs $P_{X_t1,...,X_tn}(x_1,...,x_n)$ for all t in T and all x in S.

### Markov Processes

* Markov Process: the probability of any particular future behavior of the process, when its current state is known exactly, is not altered by additional knowledge concerning its past behavior. Formally, $\operatorname{Pr}\left\{X_{n+1}=j | X_{0}=i_{0}, \ldots, X_{n-1}=i_{n-1}, X_{n}=i\right\} = \operatorname{Pr}\left\{X_{n+1}=j | X_{n}=i\right\}$ - this is the Markov property.
* One step transition probability: $P_{i j}^{n, n+1}=\operatorname{Pr}\left\{X_{n+1}=j | X_{n}=i\right\}$. When the one-step transition probabilities are independent of the time variable n, we say that the Markov chain has stationary transition probabilities (time homogeneous), ie $P_{i j}^{n, n+1}=P_{i j}$ independent of n. The probability of transitioning from one state to another is independent of time.
* Markov matrix: $\mathbf{P}=\left\|P_{i j}\right\|$ where the ith row is the probability distribution of the values of $X_{n+1}$ under the condition that $X_n = i$. All probabilities greater than 0 and all rows sum to 1
* A discrete time **Markov chain** is a stochastic process $(X_t),\;t \in T$ with T = [0,1,2,...] and state space S (generally countable in this class) that satisfies the Markov property, defined as $\forall n \geq 0, \; \forall x_0,...,x_n \in S,\;\operatorname{Pr}\left\{X_{n+1}=j | X_{0}=i_{0}, \ldots, X_{n-1}=i_{n-1}, X_{n}=i\right\} = \operatorname{Pr}\left\{X_{n+1}=j | X_{n}=i\right\}$
* Time homogeneous MC when transition probabilities are independent of n. In this case define the transition matrix of entries $P_{xy} = P(x,y):=P(X_{n+1} | X_n = x) = P(X_1=y|X_o=x)$. Note, if S is infinite then P is an infinite matrix.
* We write $\pi_x = \pi^n(x) = P(X_n=x)$: the pmf of state of chain at time n. Proposition: Distribution of time homogeneous MC is completely determined by P and $\pi_0$, the PMF of the initial state. For $n \geq 1$ and any $x_o,...,x_n \in S, \; P(X_n=x_n,...,X_0=x_0) = P(X_n=x_n|X_{n-1}=x_{n-1},...,X_0=x_0)P(X_{n-1}=x_{n-1},...,X_0=x_0)$. Then applying rules for homogeneous MC $P(X_n=x_n|X_{n-1}=x_{n-1},...,X_0=x_0) = P_{X_{n-1},X_n}$, ie. the additional history does not matter. Repeating $P_{X_{n-1}, X_n}P_{X_{n-2},X_{n-1}}..P(X_0=x_0)$ where $P(X_0=x_0) = \pi_0(x_0)$. 
* In book notation, a markov process is fully defined by its transition matrix and initial state $X_0$: $\operatorname{Pr}\left\{X_{0}=i_{0}, X_{1}=i_{1}, \ldots, X_{n}=i_{n}\right\} = p_{i_{0}} P_{i_{0}, i_{1}} \cdots P_{i_{n-2}, i_{n-1}} P_{i_{n-1}, i_{n}}$
* n-Step Probability Matrices
  * $\mathbf{P}^{(n)}=\left\|P_{i j}^{(n)}\right\|$ denotes the probability that the process goes from state i to state j in n transitions. $P_{i j}^{(n)}=\operatorname{Pr}\left\{X_{m+n}=j | X_{m}=i\right\}$
  * Theorem: $P_{i j}^{(n)}$ satisfies $P_{i j}^{(n)}=\sum_{k=0}^{\infty} P_{i k} P_{k j}^{(n-1)}$ for $P_{i j}^{(0)}=\left\{\begin{array}{ll}
    {1} & {\text { if } i=j} \\
    {0} & {\text { if } i \neq j}
    \end{array}\right.$. This gives the conclusion: $\mathbf{P}^{(n)}=\mathbf{P}^{n}$
  * For distribution of state at time n: $\pi^{(n)}(y) = \sum_{X \in S} \pi^0(x)P^{(n)}(x,y) = (\pi^0 \cdot P^n)(y)$ so $\pi^n = \pi^0P^n$ as row vectors
* Properties of Transition Matrices
  * Stochastic matrix $P(x,y) \geq 0,\; \forall x,y \in S$
  * $\sum_{y \in S}P(x,y) = 1,\;\forall x \in S$
  * Conversely any N x N stochastic matrix P and a pmf $\pi^0$ row vector give rise to a MC on {1,2,...,N} = S

### First Step Analysis

* This method proceeds by analyzing the possibilities that can arise at the end of the first transition, and then invoking the law of total probability + the Markov property to establish a characterizing relationship among the unknown variables.
* Transitory vs. Absorption Model: Transitory states in which further moves can occur and absorption states from which the model can no longer move
  * Given multiple absorption states, want to determine in which state the process gets trapped and at what time period.
  * Let $\{X_n\}$ be a finite-state Markov chain whose states are labeled 0,1,...,N. Suppose that states 0,1,..., r - 1 are transient in that $P_{i j}^{(n)} \rightarrow 0 \text { as } n \rightarrow \infty \text { for } 0 \leq i, j<r$ while states r,...,N are absorbing $\left(P_{i i}=1 \text { for } r \leq i \leq N\right)$. 
  * The transition matrix has the form $\mathbf{P}=\left[ \begin{array}{ccc}
    {\mathbf{Q}} & {\mathbf{R}} \\
    {\mathbf{0}} & {\mathbf{I}}
    \end{array}\right]$ for $Q_{i j}=P_{i j} \text { for } 0 \leq i, j<r$ 
  * The probability of ultimate absorption in state k, as opposed to some other absorbing state, depends on the initial state X0 = i. Let $U_{ik} = u_i$ denote this probability. Starting from state i, with probability $P_ik$ the process immediately goes to state k, thereafter to remain, and this is the first possibility considered. Alternatively, the process could move on its first step to an absorbing state $j \neq k$, where $r \leq j \leq N$, in which case ultimate absorption in state k is precluded. Finally, the process could move to a transient state j < r. Then $U_{i k}=P_{i k}+\sum_{j=0}^{r-1} P_{i j} U_{j k}, \quad i=0,1, \ldots, r-1$
* General Absorbing Markov Chain
  * Random absorption time T: $T=\min \left\{n \geq 0 ; X_{n} \geq r\right\}$. Let us suppose that associated with each transient state i is a rate g(i) and that we wish to determine the mean total rate that is accumulated up to absorption.
  * $w_i$, the mean total amount for starting position $X_0 = i$: $w_{i}=E\left[\sum_{n=0}^{T-1} g\left(X_{n}\right) | X_{0}=i\right]$
  * The sum $\sum_{n=0}^{T-1} g\left(X_{n}\right)$ always includes first term $g\left(X_{0}\right)=g(i)$. Proceeding from a future transient state j, $w_{i}=g(i)+\sum_{j=0}^{r-1} P_{i j} w_{j} \quad \text { for } i=0, \ldots, r-1$ 
  * $W_{ik}$, the mean number of visits to state k prior to absorption: $W_{i k}=\delta_{i k}+\sum_{j=0}^{r-1} P_{i j} W_{j k} \quad \text { for } i=0,1, \ldots, r-1$ 

### Special Markov Chains

##### 2-State Markov Chain

* $\mathbf{P}=\left\|\begin{array}{cc}
  {1-a} & {a} \\
  {b} & {1-b}
  \end{array}\right\|$ for $0 < a, b < 1$
* The n-step transition matrix: $\mathbf{P}^{n}=\frac{1}{a+b}\left[\begin{array}{ll}
  {b} & {a} \\
  {b} & {a}
  \end{array}\right]+\frac{(1-a-b)^{n}}{a+b} \left[ \begin{array}{rr}
  {a} & {-a} \\
  {-b} & {b}
  \end{array} \right]$ . Note $\lim _{n \rightarrow \infty} \mathbf{P}^{n}=\left\|\begin{array}{ll}
  {\frac{b}{a+b}} & {\frac{a}{a+b}} \\
  {\frac{b}{a+b}} & {\frac{a}{a+b}}
  \end{array}\right\|$ - The system, in the long run, will be in state 0 with probability b/(a + b) and in state 1 with probability a/(a + b), irrespective of the initial state in which the system started.

* Independent RV’s - all rows of P are identical, since $X_{n+1} \perp X_n$

##### One-Dimensional Random Walks

* Markov chain with finite or infinite state space. If the particle is in state i, it can either stay in i or move to either i+1, i-1.
* Transition matrix $P= \left[ \begin{array}{rl}
  {p_{0}} & {0 \quad \cdots} & {0} & {\cdots} \\
  {r_{1}} & {p_{1}} & {\ldots} & {0} & {\ldots} \\
  {2} & {r_{2}} & {\ldots} & {0} & {\ldots} \\
  {\vdots} & {} & {} & {} \\
  {} & {0} & {q_{i}} & {r_{i}} & {p_{i}} & {0} \\
  {} & {} & {\ddots} & {} & {} & {\ddots} \\
  \end{array} \right]$ for $p_{i}>0, q_{i}>0, r_{i} \geq 0, \text { and } q_{i}+r_{i}+p_{i}=1, i=1,2, \ldots(i \geq 1), p_{0} \geq 0, r_{0} \geq 0, \;r_{0}+p_{0}=1$
* If $X_n = i$ for i > 1, $\begin{aligned}
  \operatorname{Pr}\left\{X_{n+1}=i+1 | X_{n}=i\right\}=p_{i},\;
  \operatorname{Pr}\left\{X_{n+1}=i-1 | X_{n}=i\right\}=q_{j},\;
  \operatorname{Pr}\left\{X_{n+1}=i | X_{n}=i\right\}=r_{i}
  \end{aligned}$ 

##### Success Runs

* Transition matrix $\mathbf{P}=\left\|\begin{array}{llllll}
  {p_{0}} & {q_{0}} & {0} & {0} & {0} & {\cdots} \\
  {p_{1}} & {r_{1}} & {q_{1}} & {0} & {0} & {\cdots} \\
  {p_{2}} & {0} & {r_{2}} & {q_{2}} & {0} & {\cdots} \\
  {p_{3}} & {0} & {0} & {r_{3}} & {q_{3}} & {\cdots} \\
  {\vdots} & {\vdots} & {\vdots} & {\vdots} & {\vdots} & {}
  \end{array}\right\|$ for $q_i > 0,\; p_i > 0,\; p_i+q_i +r_r =1$
* The zero state plays a distinguished role in that it can be reached in one transition from any other state, while state i + 1 can be reached only from state i.
* Useful for applications counting successes in a row, renewal processes like lightbulb age - resets at burnout. 

### Branching Processes

* Say organisms produce $\xi$ offspring, ${Pr}\{\xi=k\}=p_{k} \quad \text { for } k=0,1,2, \ldots$
* The process $\{X_n\}$ where $X_n$ is the population size at the nth generation, is a Markov chain of special structure called a branching process.
* In the nth generation, the Xn individuals independently give rise to numbers of offspring $\xi_{1}^{(n)}, \xi_{2}^{(n)}, \ldots, \xi_{X_{n}}^{(n)}$ - the cumulative number produced for the (n + 1)st generation is $X_{n+1}=\xi_{1}^{(n)}+\xi_{2}^{(n)}+\cdots+\xi_{X_{n}}^{(n)}$
* Mean: recursively defined $M(n+1)=\mu M(n)$, generally $M(n)=\mu^{n} \quad \text { for } n=0,1, \ldots$ where $X_n$ is the population size at time n and M(n) is the mean of $X_n$
  * The mean population size increases geometrically when $\mu$ > 1, decreases geometrically when $\mu$< 1, and remains constant when $\mu$ = 1.
* Variance: recursively defined $V(n+1)=\sigma^{2} M(n)+\mu^{2} V(n)$, generally $V(n)=\sigma^{2} \mu^{n-1} \times\left\{\begin{array}{ll}
  {n} & {\text { if } \mu=1} \\
  {\frac{1-\mu^{n}}{1-\mu}} & {\text { if } \mu \neq 1}
  \end{array}\right.$ 
  * The variance of the population size increases geometrically if $\mu$ > 1, increases linearly if $\mu$ = 1, and decreases geometrically if $\mu$ < 1

##### Extinction

* The random time of extinction N is thus the first time n for which Xn = 0, and then, obviously, Xk = 0 for all $k \geq N$.
* In Markov chain terminology, 0 is an absorbing state, and we may calculate the probability of extinction by invoking a first step analysis.
* Defn $u_{n}=\operatorname{Pr}[N \leq n]=\operatorname{Pr}\left\{X_{n}=0\right\}$, be the probability of extinction at or prior to the nth generation, beginning with a single parent X0 = 1. The k subpopulations independent, w/ original statistical properties. Each has probability of dying out in n-1 generations equal to $u_{n-1}$. 
* Probability that all die out in n-1 generations is $u_{n-1}^k$ by independence. Then weighting by probability of k offspring:
* $u_{n}=\sum_{k=0}^{\infty} p_{k}\left(u_{n-1}\right)^{k}$

### Class Examples

1. Stochastic Processes - Simple random walk on Z
   * $T = \{0,1,2,...\},\;S = Z$. Given the history of the walk, we derive the n + 1 position: $P(X_{n+1} = k | X_0=x_0,....,X_n=x_n) = P(X_{n+1} = k | X_n = x_n)$. This is the Markov property, $=\begin{cases}1/2 & k=X_n+1\;or\;X_n-1 \\ 0 & else\end{cases}$. Gives joint PMFs pf $(X_0,...,X_n)$ for any n, get PMFs $(X_{t1},...,X_{tn})$ by summing out the extra variables. 
   * Ex - $P_{X_0,X_2}(x_0,x_2) = \sum_{x_1}P_{X_0,X_1,X_2}(x_0,x_1,x_2)$ - summing over the probabilities for x1 gives you the marginal desired.
2. Galton Watson Branching Process / Tree
   * $X_t$ = size of the family at generation t. Say X0 = 1, starting with one individual. S= [0, 1, 2, ...] = T. 
   * Specifying the distributions: joint distributions of $X_{t1},...,X_{tn}$ determined recursively by $X_{t+1} = \sum_{k=1}^{X_t} Y_{t,k}$. Define $Y_{t,1},...,Y_{t,X_t} \sim_{iid} Pois(\lambda)$ conditional on $X_t$; the Y's are the number of children each person has at a generation, where X's are the state of the family at time t. Note another sum up to a RV up to Xt. 
     * Could alternatively draw $\{Y_{t,k}\}^{\infty,\infty}_{t=0,k=1}$ iid Pois beforehand, but would not use a lot of those variables.
3. Gambler’s Ruin
   * Game where we win in each round (independent) 1 dollar with probability p=0.4 and lose one dollar with probability 1-p=0.6. Decide ahead of time to quit once we reach N dollars. The game is over once we reach 0 dollars. $X_n$= amount of money we have after n rounds. State space S = {0,1,...,N} - between and including the absoption states.
   * If still playing $X_n \in [1,N-1],\; P(X_{n+1} = x_n+1|X_n=x_n,...,X_0=x_0) = 0.4 \text{ and }P(X_{n+1} = x_n-1|X_n=x_n,...,X_0=x_0) = 0.6$
   * $P_{0,0}=1,\; P_{N,N} =1$ - if you start at either end state the game is immediately over. At every other row, have (0.6,0,0.4) centered at the column equal to the row position. Alternatively, could create a directed graph of the possible state transitions with edge weights equal to the probabilities with that state transition - this makes P an adjacency matrix.
   * Now ask what are the chances of falling into 0 or N and how long does it take to get there? Given $X_0 = x\in\{1,2,3\}$
   * $P = \left[\begin{array}{c}1 & 0 & 0 & 0 \\0.6 & 0 & 0.4 & 0\\0 & 0.6 & 0 & 0.4\\...\end{array}\right]$ Computer calculations suggest $lim_{n\rightarrow \infty}(P^n) = \left[\begin{array}{c}1\\57/65\\45/65\\27/65\\0\end{array}...0...\begin{array}{c}0\\8/65\\20/65\\38/65\\1\end{array}\right]$ with zeroes for all middle columns.
4. Ehrenfest Chain
   * Stat Physics model for two equal sized containers of gas connected by a small opening. Expect equilibrium eventually with same number of molecules in each container (balls, urns).
   * N = total # of balls (order 10^23), particle exchange modeled as a random process, pick 1 ball uniformly at random a move to the other urn. Let $X_n = $# of balls in the left urn after the nth draw, change +- 1 at each step.
   * $P(X_{n+1} = x + 1 | X_n=x) = \frac{N-x}{N}$ = # balls in right urn / total N. $P(X_{n+1} = x - 1 | X_n=x) = \frac{x}{N}$ = # balls in left urn / total N. Note observes the markov property.
   * 1) How long until $x_n \approx N/2$? 2) Does it stay there? 3) How much does it fluctuate? 4) How often does the chain reach endpoints 0,N?
   * For N=4, S = [0,1,2,3,4], $P=\left[\begin{array}{ccccc}0&1&0&0&0\\1/4&0&3/4&0&0\\0&1/2&0&1/2&0\\0&0&3/4&0&1/4\\0&0&0&1&0\\\end{array}\right]$. Similar structure to Gambler’s ruin, except:
     * Probability of moving right / left not constant
     * 0,4 are now repelling instead of absorbing states. 
     * Long term behavior should be very different - gravitate to the middle rather than the endpoints
5. Simple Random Walk (SRW)
   * Let G = (V, E) be a graph. ||V|| = 6, E=[[1,2], [3,4], [3,5], [4,5], [5,6]]. 
   * SRW on G: state space S = V, transition probabilities $P(x,y) = \begin{cases}1/degree(x) & for\; \{x,y\}\in E\\0 & otherwise\end{cases}$ - that is at each time step move along an edge at random chosen uniformly at random from set of edges leading out of current state.
   * For V finite, is the distribution of $X_t \approx uniform$ on V after a long time? For our example, certainly not since 1-2 and 3-4-5-6 are separate CCs, whichever component contains $X_0$ contains $X_t\; \forall t \geq 0$. Additionally, vertices have different degrees, leading to different probabilities of visiting those nodes. Can come up with less intuitive examples as well, such as a connected 2D square - only can visit some vertices on odd states, others on even states.
   * Recalls the drunkard on the street, that is a SRW on G with V = Z and nearest neighbor edges $E = \{\{k,k+1\}:k\in \Z\}$
6. iid Sequence
   * Let $\{X_n\}_{n=0}^\infty$ be iid RVs taking values in a countable set S with distribution $p(x) = P(X_n=x)$ then $P(X_{n+1}=x|X_n=x,...,X_0=x_0)=P(X_{n+1}=x|X_n=x) = P(X_{n+1}=x) =p(x)$
   * Transition matrix $P_{X,Y} = P(X_1=y)= p(y)$ - P has identical rows all equal to the row vector p
7. Deterministic Chain
   * MC with $X_t$, S = [1,2,3,4] and P(1,2) = P(2,3) = P(3,4) = P(4,1) = 1 and P(x,y) = 0 - think of moving around edges of a box
   * $X_t$ is completely deterministic conditional on $X_0$ but it is still an MC
   * Color state 1 red, 2,4 green, 3 blue. Define $Y_t = $ color of $X_t$. Is Y an MC? No, consider if previous color is green, we may be going to red or blue with probability 1 depending on if the color prior to green was red or blue, ie $P(Y_2 =blue|Y_1=green,Y_0 = red) = 1 \neq 0 = P(Y_2 =blue|Y_1=green,Y_0 = blue)$ Distribution of the next state depends on both current and previous states. The fact that MC is deterministic did not cause this problem though.


## Probability Reference

### Probabilities of Events

* Intersection - probability that both A and B occur
* Complement - $A^c$ event that A does not occur, all events in the sample space that are not A
* Disjoint - A and C are disjoint if $A \cap C = \empty$ 
* Probability Axioms: 1) $P(\Omega) = 1$, 2) If $A \subset \Omega$ then $P(A) \geq 0$ 3) If A, B disjoint then $P(A \cup B) = P(A) + P(B)$
* Addition Law: $P(A \cup B) = P(A) +P(B) - P(A \cap B)$
* Permutation: ordered arrangement of objects
* Binomial coefficients: $(a+b)^n = \sum_{k=0}^n a^kb^{n-k}$
* \# of ways n objects can be grouped into r classes with $n_I$ in the $i^{th}$ class: ${n\choose n_1n_2...n_r} = \frac{n!}{n_1!n_2!...n_r!}$
* Bayes, multiplication law: $P(A|B) = \frac{P(A \cap B)}{P(B)}$,  $P(B_j|A) = \frac{P(A|B_j)P(B_j)}{\sum P(A|B_i)P(B_i)}$
* Law of total probability: $P(A) = \sum P(A|B_i)P(B_i)$
* Independence for sets: $P(A \cap B) = P(A)P(B)$. Mutual independence implies pairwise independence

### Probability Algebra / Calculus

* Change of variables: $Y=g(X), \; F_{Y}(y)=\operatorname{Pr}\{Y \leq y\}=\operatorname{Pr}\left\{X \leq g^{-1}(y)\right\}=F_{X}\left(g^{-1}(y)\right)$. Then $f_{Y}(y)=\frac{1}{g^{\prime}(x)} f_{X}(x), \text { where } y=g(x)$

##### Hazard + Survival Functions

* Survival Function: $S(t)=P(T>t)=1-F(t)$. Simply a reversal of the CDF for data consist of time until death or failure, chance of surviving past t.
* Hazard: As the instantaneous death rate for individuals who have survived up to a given time. $h(t)=\frac{f(t)}{1-F(t)} = \frac{f(t)}{S(t)} = -\frac{d}{dt}log(S(t))$
* May be thought of as the instantaneous rate of mortality for an individual alive at time t. If T is the lifetime of a manufactured component, it may be natural to think  of h(t) as the instantaneous or age-specific failure rate.

##### Convolutions

* X, Y independent RVs, Z = X + Y, then: $F_{Z}(z)=\int_{-\infty}^{+\infty} F_{X}(z-\xi) \mathrm{d} F_{Y}(\xi)=\int_{-\infty}^{+\infty} F_{Y}(z-\eta) \mathrm{d} F_{X}(\eta)$
* With PDFs: $f_{Z}(z)=\int_{-\infty}^{\infty} f_{X}(z-\eta) f_{Y}(\eta) \mathrm{d} \eta=\int_{-\infty}^{+\infty} f_{Y}(z-\xi) f_{X}(\xi) \mathrm{d} \xi$
* With nonnegative X and Y: $f_{Z}(z)=\int_{0}^{z} f_{X}(z-\eta) f_{Y}(\eta) \mathrm{d} \eta=\int_{0}^{z} f_{Y}(z-\xi) f_{X}(\xi) \mathrm{d} \xi \quad \text { for } z \geq 0$

### Expectations, Moments, Variances

* Expectation: $E(X) = \begin{cases} \sum_x xp_X(x) & \text{discrete} \\ \int_{-\infty}^\infty xf_X(x) dx & \text{continuous}\end{cases}$. Weighted average of x values by probabilities
* Linearity of expectation: $E(aX + bY) = aE(X) + bE(Y)$
* $\operatorname{Var}(X)=E\left\{[X-E(X)]^{2}\right\}=E\left(X^{2}\right)-[E(X)]^{2}$
* MGF uniquely determines a probability distribution - same MGF means same distribution
* $M(t) = E(e^{tX})= \int_{-\infty}^\infty e^{tx} f(x) \, dx$ 
* The rth moment: $M^{(r)}(0) = E(X^r)$
* $\text { If } X \text { has the } \operatorname{mgf} M_{X}(t) \text { and } Y=a+b X, \text { then } Y \text { has the mgf } M_{Y}(t)=e^{a t} M_{X}(b t)$
* If X, Y independent and Z= X + Y, $M_z(t) = M_x(t)M_y(t)$
* mth central moment - the mth moment of the RV $X-\mu x$. Variance is the second central moment. The first central moment is 0

###### Markov Inequality

* $X \geq 0$, then $P(X \geq t) \leq \frac{E(X)}{t}$
* This result says that the probability that X is much bigger than E(X) is small. 

###### Chebyshev’s Inequality

* $P(|X-\mu|>t) \leq \frac{\sigma^{2}}{t^{2}}$
* Plug in t = K x sigma to get alternate form. Not necessary to bound X to be positive.

### Probabilty Distributions

##### Discrete Mass Functions

* Bernoulli: $p\left (x \right ) = \begin{Bmatrix}
  p^{x} \left ( 1-p \right )^{1-x}, & if \: x = 0 \: or \: x = 1
  \\ 
  0, & otherwise
  \end{Bmatrix}, 
  \: 0\leq p\leq 1$
* Binomial: $p(k)=\binom{n}{k}p^{k}(1-p)^{n-k}\\
  n \in \mathbb{N}, k=0,1,...,n, \,
  0\leq p\leq 1$
  * $Bin(n,p) = \sum_{i=1}^n X_i$, for X Bern(P)
  * $E(X) = np, Var(x) = np(1-p)$
  * Can view in terms of Bern: $X =^d \sum_{i=1}^n X_i, \; X_i \; iid \; Ber(p)$
* Geometric - # trials until success
  *  $p(k)=p(1-p)^{k-1}, \quad k=\N$
  * $\begin{aligned} E(X) &=\frac{1}{p} \\ \operatorname{Var}(X) &=\frac{1-p}{p^{2}} \end{aligned}$
  * From Bernoulli: $X_1,X_2,...\;iid \; Ber(p)$, then $X = ^d min\{n: X_n =1\}$ (min of n st X = 1 / heads)
* Geometric - # failures prior to success
  * $p(k)=p(1-p)^{k}$, $k = 0 \cup \N$
  * $E[Z]=\frac{1-p}{p} ; \quad \operatorname{Var}[Z]=\frac{1-p}{p^{2}}$
* Negative Binomial Using Geom 1: $P(X=k) = {k-1 \choose r-1}p^r (1-p)^{k-r}
  \quad \quad
  , 0 \leq p \leq 1,  k = r, r+1, \ldots,  r = 1, 2, \ldots, k$
* Negative Binomial Using Geom 2 (failures before rth success): $p(k)=\operatorname{Pr}\left\{W_{r}=k\right\}=\frac{(k+r-1) !}{(r-1) ! k !} p^{r}(1-p)^{k}$, for $k \in 0 \cup \N$ 
* Hypergeometric

  * *n*: population size; *n*∈N
  * *r*: successes in population; *r*∈{0,1,...,*n*}
  * *m*: number drawn from population; *m*∈{0,1,...,*n*}
  * *X*: number of successes in drawn group
  * $P(X=k)=\frac{\binom{r}{k}\binom{n-r}{m-k}}{\binom{n}{m}} \, \max(0,m+r-n) \leq k \leq \min(r,m) \, 0 \leq p(k) \leq 1$
* Poisson: $P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda}\, k = 0,1,2,3,...,\,\, \lambda > 0$
  * $E(X) = \lambda, Var(X) = \lambda$
  * $e^-\lambda$ is the normalizing factor, since $\sum \frac{\lambda^k}{k!}$ is the power series for $e^\lambda$
  * Law of small numbers: The binomial distribution with parameters n and p converges to the Poisson with parameter $\lambda$ if $n \rightarrow \infty$ and $p\rightarrow 0$ in such a way that $\lambda = np$ remains constant. In words, given an indefinitely large number of independent trials, where success on each trial occurs with the same arbitrarily small probability, then the total number of successes will follow, approximately, a Poisson distribution. $Bin(n, \frac{\lambda}{n}) \rightarrow_d Pos(\lambda)$
* Multinomial: $\operatorname{Pr}\left\{X_{1}=k_{1}, \ldots, X_{r}=k_{r}\right\} = \left\{\begin{array}{ll}
  {\frac{n !}{k_{1} ! \cdots k_{r} !} p_{1}^{k_{1}} \cdots p_{r}^{k_{r}}} & {\text { if } k_{1}+\cdots+k_{r}=n} \\
  {0} & {\text { otherwise }}
  \end{array}\right.$
  * $E\left[X_{i}\right]=n p_{i}, \operatorname{Var}\left[X_{i}\right]=n p_{i}\left(1-p_{i}\right), \; \operatorname{Cov}\left[X_{i} X_{j}\right]=-n p_{i} p_{j}$

##### Continuous Density Functions

* Uniform: $f(x) = \begin{cases}   1/ (b - a) & a \leq x \leq b\\  0 & x < a \ or \ x > b \\  \end{cases} \,, x \in [a,b] \,,  a<b\, , f : \mathbb{R} \mapsto [0, \infty )$
  * $E(X) = \frac{1}{2}(a+b)$
  * $Var(X) = \frac{1}{12}(b-a)^2$
* Exponential $f(x) = \left\{\begin{matrix}   \lambda e^{-\lambda x} , &  x \geq 0\\   0 , & x< 0 \end{matrix}\right. \, ,\, \lambda > 0$
  * $E(X) = \frac{1}{\lambda}$
  * $Var(X) = \frac{1}{\lambda^2}$
* Normal: $f(x\mid \mu,\sigma^2)=\frac{1}{\sqrt{2 \pi}\sigma }e^{-\frac{(x-\mu)^2}{2\sigma^2}}, f:\mathbb{R}\to(0,\infty), \mu\in \mathbb{R},\sigma>0$
* Log Normal $V=e^{X}, X \sim N$: $f_{V}(v)=\frac{1}{\sqrt{2 \pi} \sigma v} e^{-\frac{1}{2}\left(\frac{\ln v-\mu}{\sigma}\right)^{2}}, \quad v \geq 0$ 
  * $E[V]=e^{\mu+\frac{1}{2} \sigma^{2}}$
  * $\operatorname{Var}[V]=\exp \left\{2\left(\mu+\frac{1}{2} \sigma^{2}\right)\right\}\left[\exp \left\{\sigma^{2}\right\}-1\right]$
* Gamma: 
  * $\Gamma(x) = \int_0^\infty u^{x-1}e^{-u} \, du, x > 0$
  * $g(t \mid \alpha, \lambda) = \begin{cases} \frac{\lambda^\alpha}{\Gamma(\alpha)} t^{\alpha-1} e^{-\lambda t}, t \geq 0 \\ 0, t < 0 \end{cases}$
  * $g: \mathbb{R} \to [0,\infty), \alpha > 0, \lambda > 0$
  * $\begin{aligned} E(X) &=\frac{\alpha}{\lambda} \\ \operatorname{Var}(X) &=\frac{\alpha}{\lambda^{2}} \end{aligned}$ 
* Beta: $f(u) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}u^{a-1}(1-u)^{b-1}, 0\leq u\leq 1, \,\, a, b > 0$ 
  * $\begin{aligned} E(X) &=\frac{a}{a+b} \\ \operatorname{Var}(X) &=\frac{a b}{(a+b)^{2}(a+b+1)} \end{aligned}$
* Cauchy: $f_Z(z) = \frac{1}{\pi(z^2 + 1)}$ for $z \in (-\infty,\infty)$

### Weak Law of Large Numbers

* $X_1 ... X_i \sim iid$, $E(X_i) = \mu $ and $ Var(X_i) =\sigma^2$ . Let $X_n = n^{−1} \sum_{i=1}^n X_i$ and 
* $lim_{n \rightarrow \infty}P(|\bar{X_n} - \mu| > \epsilon) = 0$
* Convergence in probability

### Central Limit Theorem

* For $X_i$ iid, $E(X) =\mu$,   $Var(X)=\sigma^2$
* $\lim _{n \rightarrow \infty} \operatorname{Pr}\left(\frac{\bar{X}_{n}-\mu}{\sigma / \sqrt{n}}  \leq x\right)=\Phi(x)$
* Convergence in distribution

### Useful Functions and Integrals

* Gamma
  * $\Gamma(\alpha)=\int_{0}^{\infty} x^{\alpha-1} \mathrm{e}^{-x} \mathrm{d} x, \quad \text { for } \alpha>0$
  * $\Gamma(\alpha)=(\alpha-1) \Gamma(\alpha-1)$
  * $\Gamma(k)=(k-1) ! \quad \text { for } k=1,2, \ldots$
  * $\Gamma\left(\frac{1}{2}\right)=\sqrt{\pi}$
* Stirling’s Formula: 
  * $n !=n^{n} \mathrm{e}^{-n}(2 \pi n)^{1 / 2} \mathrm{e}^{r(n) / 12 n}$ in which $1-\frac{1}{12 n+1}<r(n)<1$
  * More loosely, $n ! \sim n^{n} \mathrm{e}^{-n}(2 \pi n)^{1 / 2} \quad \text { as } n \rightarrow \infty$
  * Implies binomial coefficients $\left(\begin{array}{l}
    {n} \\
    {k}
    \end{array}\right) \sim \frac{(n-k)^{k}}{k !} \quad \text { as } n \rightarrow \infty$
* Beta
  * $B(m, n)=\int_{0}^{1} x^{m-1}(1-x)^{n-1} \mathrm{d} x = \frac{\Gamma(m) \Gamma(n)}{\Gamma(m+n)} \quad \text { for } m>0, n>0$
  * $B(m+1, n+1)=\int_{0}^{1} x^{m}(1-x)^{n} \mathrm{d} x=\frac{m ! n !}{(m+n+1) !}$
* Geometric Series
  * $\sum_{k=0}^{\infty} x^{k}=1+x+x^{2}+\cdots=\frac{1}{1-x} \quad \text { for }|x|<1$
* Sums
  * $1+2+\cdots+n=\frac{n(n+1)}{2}$
  * $1+2^{2}+\cdots+n^{2}=\frac{n(n+1)(2 n+1)}{6}$
  * $1+2^{3}+\cdots+n^{3}=\frac{n^{2}(n+1)^{2}}{4}$