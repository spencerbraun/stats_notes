---
title: Stochastic Processes II
date: 20200407
author: Spencer Braun
---

[TOC]

# Stochastic Processes II

## Introduction

* We have an experiment -> set of possible outcomes called the sample space $\Omega$
* An event is a subset of outcomes
* Axioms of probability
	* $0 \leq P(E) \leq 1$ for all events E (P(E) = probability of E)
	* $P(\Omega) = 1$
	* If $E_1,E_2,...$ disjoint events, then $P(\cup_{i=1}^\infty E_i) = \sum_{i=1}^\infty P(E_i)$. Disjoint means $E_i \cap E_j = \empty$ whenever $i\neq j$. Probability of the union is the sum of probabilities for disjoint events.
* RV is a function from $\Omega$ into $\R$. 
* A stochastic process is a collection of RVs $(X_t)_{t \in T} =X$. T can be $\{1,2,3,...\}$ or $[0,\infty)$; a sequence or continuum of RVs.

## Chapter 3: Renewal Theory
* Let $X_1,X_2,...$ be iid nonnegative RVs. Assume $P(X_n = 0 ) < 1$, there is some chance that our RV is nonzero. Let $\mu = E(X_n)$ where we allow $\mu$ to be $\infty$, but we have a well-defined expected value.
	* For example, if X is integer-valued and $P(X = k) = \frac{c}{k^2}$ where $c \sum \frac{1}{k^2} = 1$ then $E(X) = \infty$
* These variables are called **interarrival times**. Imagine clients arriving at a server, $X_i =$ time between $(i-1)th$ and ith arrival. Clients are immediately served (no queueing involved). Let $S_0= 0, \; S_n = \sum_{i=1}^n X_i$. Then $S_n=$ arrival time of the nth client.
* Let $N(t) = sup\{n: S_n \leq t\}$ (sup since theoretically there could be infinitely many) = number of clients arriving by time t. Maximum n such that the nth client has arrived by time t.
* The stochastic process $(N(t))_{t \geq 0}$ is called a **renewal process**. Since the interarrival times are iid, it follows that at each rewnwal the process probabilistically starts over. Think of replacing lightbulbs - it stays on for a certain time before being replaced. N(t) is the number of bulbs you have to replace by time t. Plotted, we see piecewise constant function, jumping to the next integer i at each $S_i$. The times at which the jumps happen are random. A Poisson process is an example of a renewal process, limited to interarrival times with exponential random variables.
* Let $F_n(t) = P(S_n \leq t)$. This function is a CDF of a sum of RVs $S_n$, may not be easy to evaluate but can still be quite useful. Observe that $N(t) \geq n \iff S_n \leq t$; the number of variables up to the time t iff the nth customer has arrived by time t. Therefore $P(N(t) =n) = P(N(t) \geq n) - P(N(t) \geq n + 1) = P(S_n \leq t ) - P(S_{n+1} \leq t) = F_n(t) - F_{n+1}(t)$. If we know the CDF then we can calculate the probabilities for N(t).
* **Renewal function**: expected number of variables up to time t. The renewal function $m: [o, \infty)\rightarrow \R$ is defined as $m(t) = E(N(t) )=$ expected number of arrivals by time t.
* Generally we observe the renewal process for a while, observe the distribution of interarrival times then try to say something about what we can expect in the future.
* <u>**Theorem**</u> (Proposition 3.2.1, 3.2.2): $m(t) = \sum_{n=1}^\infty F_n(t)$ and $m(t) < \infty$ for all t.
  * We can get F from the interarrival times, then calculate m and get the expected number of arrivals in some interval i in the future. m(t) is finite no matter what distribution we use. 
  * **Lemma 1**: Let X be a non-negative integer valued RV with $E(X) < \infty$ then $E(X) = \sum_{n=1}^\infty P(X \geq n)$
    * Proof: $E(X) = P(X = 1) + 2P(X=2) + 3P(X=3)$. Regrouping $=[P(X=1) + P(X=2) + ...] + [P(X=2) + P(X=3) + ...]+[P(X=3) + P(X=4) + ...] + ...$ This sum is equal to $\sum_{n=1}^\infty P(X \geq n)$. When summing non-negative quantities, we can rearrange the sum in any way and still get the same answer.
  * **Lemma 2** (Markov's Inequality): Let X be a non negative RV with $E(X) < \infty$ then for any $t > 0,\; P(X\geq t) \leq \frac{E(X)}{t}$
    * The LHS of the inequality is often much more complicated that the RHS, so this comes in handy
    * Proof: Let y be the following RV: $y = \begin{cases} 1 & if \;x \geq t \\ 0 & else \end{cases}$. Let Z = X/t. Then if X < t we have $y = 0 \leq Z$ since Z is a nonnegative RV. If $X \geq t$ then $y = 1 \leq x / t = Z$. So y is always $\leq Z \implies E(y) \leq E(Z)$. Finally, $E(y) = 0P(y = 0) + 1P(y=1) = P(X \geq t)$ and $E(Z) = \frac{E(X)}{t}$
  * **Lemma 3**: Let X be a non-negative RV st $P(X = 0) < 1$, then $E(e^{-X}) < 1$.
  	*  Note if X were always 0 then $e^{-X}$ would always be 1. 
    * Proof: Suppose $E(e^{-X}) = 1$ then $E(1-e^{-X}) = 0$. Since X is nonnegative then $1 - e^{-X} \geq 0$. So  $1 - e^{-X}$ is a nonnegative RV whose expectation is 0. So by Markov's Inequality, $P(1 - e^{-X} \geq t) \leq 0$ for any $t > 0 \implies P(1 - e^{-X} =0) = 1\implies P(X = 0)=1$. So we get a contradiction. Thus $E(e^{-X}) \neq 1$. But $e^{-X} < 1$ always so $E(e^{-X})$ cannot be > 1 thus  $E(e^{-X}) < 1$
  * **Lemma 4**: $m(t) < \infty$ for any t
    * Proof: $P(N(t) \geq n) = P(S_n \leq t)$ by above. $= P(e^{-S_n} \geq e^{-t}) \leq \frac{E(e^{-S_n})}{e^{-t}}$. Using the fact that X's are iid, then $ = e^t (E(e^{-X_1})^N) = e^tp^n$. By Lemma 1 $p = E(e^{X_1})$. Then $m(t) = E(N(t)) = \sum_{n=1}^\infty P(N(t) \geq n) = \sum_{n=1}^\infty P(S_n \leq t = \sum_{n=1}^\infty) F_n(t)$. By the above $\sum_{n=1}^\infty P(N(t) \geq n) \leq e^t \sum_{n=1}^\infty p^n < \infty$ since $0 \leq p < 1$.
* Note the above also shows $E\left[N^{r}(t)\right]<\infty$ for all $t ,r \geq 0$
* Recall: $\mu = E(X_1) = E(X_2)=...$ which could be infinite.
* **<u>Theorem</u>** (Proposition 3.3.1): With probability 1, $\underset{t\rightarrow \infty}{lim} \frac{N(t)}{t} = \frac{1}{\mu}$
  * We will use the Strong Law of Large Numbers (SLLN) which says that $\frac{S_n}{n} \rightarrow \mu$ with probability 1. Means that the $P(\underset{t\rightarrow \infty}{lim} \frac{S_n}{n} = \mu) = 1$. This holds for mu finite or infinite.
  * **Lemma**: $\underset{t\rightarrow \infty}{lim} N(t) = \infty$ with probability 1.
  	* As we take larger times, the number of arrivals goes to infinity. The only way in which $N(\infty)$, the total number of renewals that occurs, can be finite is for one of the interarrival times to be infinite (the next arrival never occurs).
  	* Proof: The limit always exists because N is an increasing process. $\underset{t\rightarrow \infty}{lim} N(t) < \infty \iff X_n = \infty$  for some n. This is an integer valued process, so finite if it stops at some point. $P(X_n = \infty) = 0$ so $P(\underset{t\rightarrow \infty}{lim} N(t) < \infty) = p(\cup_{n=1}^\infty \{X_n = \infty\}) \leq \sum_{n=1}^\infty P(X_n = \infty) = 0$ (Probability of union is bounded by sum of probabilities). Fact: for any events $A_1,A_2,...\; P(\cup_{n=1}^\infty A_i )\leq \sum_{n=1}^\infty P(A_i)$
  * Proof: Note $S_{N(t)} \leq t < S_{N(t) + 1} $. The customer we count up to for N(t) arrives by t but the next customer must be after t by the definition of N(t). Recall $S_n$ is the arrival time of the nth client, N(t) = # of arrivals by time t. This implies $\frac{S_{N(t)}}{N(t) } \leq \frac{t}{N(t)} < \frac{S_{N(t) + 1}}{N(t)}$. $\frac{S_n}{N(t)}$ is the average of the first N(t) interarrival times, and by using our SLLN fact that $N(t) \rightarrow \infty,\; \frac{S_n}{n} \rightarrow \mu$ as $n \rightarrow \infty$ we get  $\frac{S_{N(t)}}{N(t)} \rightarrow \mu$ as $t \rightarrow \infty$. This all happens with probability 1.
  * Then $\frac{S_{N(t)}}{N(t)} = \frac{N(t) + 1}{N(t)}\frac{S_{N(t)}}{N(t) + 1} \rightarrow 1\times \mu $ as $t \rightarrow \infty$. Therefore $\frac{t}{N(t)} \rightarrow \mu$ as $t \rightarrow \infty$
* Note $S_{N(t)}$ is the time of the last renewal prior to or at time t. Therefore $S_{N(t) + 1}$ is the time of the first renewal after time t.
* $\frac{1}{\mu}$ is the rate of the renewal process. By Prop 3.3.1, with probability 1 the long-run rate at which renewals occur is equal to $\frac{1}{\mu}$. 
* **Stopping Time**: Let $X_1,X_2,...$ denote a sequence of RVs. An integer-valued RV N is said to be a stopping time for the sequence if the event $\{N=n\}$ is independent of $X_{n+1},X_{n+2},...$ for all n and N is positive integer valued.  Said another way, the occurrence or non-occurrence of event $\{N=n\}$ is completely determined by the values of $X_1,X_2,...$ for every n. 
  * Note we allow $N = \infty$ also. Whether you stop or not, our evaluation of whether to stop is completely deterministic at time n.
  * Example: Let N = $min\{n: X_1+...+X_n \geq 5\}$. First n so the sum is at least 5. Then the event $\{N = n\}$ can be rewritten as the event $\{X_1 < 5, X_1+X_2 < 5,..., X_1+...+X_{n-1} < 5, X_1+...+X_{n} \geq 5\}$ . Up to n-1, our sum is always less than 5, then at n, the sum is greater or equal to 5. Rewritten this way, it is clear that the values of $X_1,...,X_n$ completely determine whether $\{N=n\}$ has happened or not, making N a stopping time for this sequence of X's.
* **<u>Theorem 3.3.2: Wald's Equation</u>**: If $X_1,X_2,...$ are iid RVs having finite expecations ($E|X_i|<\infty$), and if N is a stopping time for $X_1,X_2,...$ st $E(N) < \infty$ then $E\left[\sum_{i}^{N} X_{n}\right]=E[N ] E[X_1]$
  * Normally with a finite sum, the expectation would just move inside. Here we have a finite sum, but the range of the sum in random (see 217 for theorems on random sums). 
  * Fact: If $Y_1,Y_2,...$ are RVs st $\sum_{i=1}^\infty E|Y_i|  < \infty$ then $E(\sum_{i=1}^\infty Y_i) = \sum_{i=1}^\infty E(Y_i)$ (Consequence of dominated convergence theorem)
  * Proof of Wald: Let $I_n = \begin{cases} 1 & if \; N \geq n \\ 0 & else \end{cases}$ . Let $Y_n = X_n I_n = \begin{cases} X_n& if \; N \geq n \\ 0 & else \end{cases}$. Then $\sum_{n=1}^N X_n = \sum_{n=1}^\infty Y_n$ since $Y_n =0 $ for $ N \geq n$. We will show that $\sum_{n=1}^\infty E|Y_n| < \infty$ then we will conclude that $E(\sum_{n=1}^N X_n) = E(\sum_{n=1}^\infty Y_n) = \sum_{n=1}^\infty E(Y_n)$. Finally we will show that $\sum_{n=1}^\infty E(Y_n) = E(N)E(X_1)$
    * First note that the value of $I_n $ is determined by the occurrence or non-occurrence of the event $\{N < n\}$. This event $\{N < n\} = \{N =1\} \cup \{N =2\} \cup ...\cup \{N =n-1\}$. The occurrence or non-occurrence of each of the events on the right can be determined by the values of $X_1,...,X_{n-1}$. Therefore $I_n$ itself is a function of $X_1,..,X_{n-1}$.  Since these variables are independent, we conclude that $X_n,I_n$ are independent. 
    * So $E|Y_n| = E|X_nI_n| = E(|X_n||I_n|) = E|X_n| E(I_n) \underset{iid}{=} E|X_1| E(I_n)$. Then $=E|X_1|P(N \geq n)$. So in summation, $\sum_{n=1}^\infty E|Y_n| = E|X_1|\sum_{n=1}^\infty P(N \geq n) = E|X_1|E(N) < \infty$. 
    * Thus $E(\sum_{n=1}^N X_n) = E(\sum_{n=1}^\infty Y_n) = \sum_{n=1}^\infty E(Y_n)$
    * But again, $E(Y_n) = E(X_n I_n) \overset{\perp}{=} E(X_n)(I_n) = E(X_1) P(N \geq n) $ So $\sum_{n=1}^\infty E(Y_n) = E(X_1) \sum_{n=1}^\infty P(N\geq n) = E(X_1) E(N)$
  * Example: Stopping time $E(N) = \infty$ and Wald's equation fails. Let $X_1,...$ iid with $P(X_1=1) = P(X_1 = -1) = 1/2$. Let $S_n = \sum_{i=1}^n X_i$, $S_0 = 0$. let $N = min\{n: S_n = 1\}$. This is a random walk, looking for first time sum equal to 1. The event $\{N=n\} = \{S_1 \neq 1,....,S_{n-1} \neq 1, S_n = 1\}$, so N is a stopping time but Wald's equation does not hold because $E(N)E(X_1) = 0$ since $E(X_1) = 0$ but $E(\sum_{i=1}^N X_i) = E(S_N) = 1$ since $S_N = 1$. We can prove that the sum will definitely hit 1 at some point. 
* **<u>Theorem (Elementary Renewal Theorem)</u>**: $\underset{t \rightarrow \infty}{lim} \frac{m(t)}{t} = \frac{1}{\mu}$
  * Note that $ \frac{m(t)}{t} = E( \frac{N(t)}{t})$, so it looks like we have this result from above. However, $X_n \rightarrow a $ does not imply $E(X_n)\rightarrow a$ so we must prove this result.
  * This tells us about the behavior of the expected number of variables. Blackwell's renewal theorem is a more useful version of this. 
  * **Lemma**: $X_1,X_2,...$ iid interarrival times. Take any $ t \geq 0$ then $N(t) + 1$ is a stopping time with respect to the sequence of X's.
    * Proof: $\{N(t) + 1 = n\} = \{N(t) = n-1\} = \{X_1+...X_{n-1} \leq t, X_1+...X_{n} > t\}$. The nth variable happens after time t, but the n-1st variable happens before or at t.
  * **Corrolary**: $E(S_{N(t) + 1}) = \mu(m(t) + 1)$ if $\mu < \infty$
    * Proof: We have shown that $E(N(t) + 1) = E(N(t) + 1) = m(t) + 1 < \infty$. So by Wald's equation, $E(S_{N(t) + 1}) = E(\sum_{i=1}^{N(t) + 1} X_i) = E(N(t) + 1)E(X_1) = \mu(m(t) + 1)$
  * Proof: First we will show that $\underset{t \rightarrow \infty}{liminf} \frac{m(t)}{t} \geq \frac{1}{\mu}$. If $\mu < \infty$ then by the corr, $\frac{\mu(m(t) + 1)}{t} = \frac{E(S_{N(t) + 1 })}{t}$, the arrival time of the N(t) + 1 customer over t, which we know have to be bigger than 1 since the arrival always happens after t. But $S_{N(t) + 1} > t \implies E(S_{N(t) + 1} ) > t$ always. This implies $\frac{\mu(m(t) + 1)}{t} > 1 \implies \underset{t \rightarrow \infty}{liminf} \frac{m(t)}{t} \geq \frac{1}{\mu}$. If $\mu = \infty$, then this is trivially true.


