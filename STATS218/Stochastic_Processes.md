---
title: Stochastic Processes II
date: 20200407
author: Spencer Braun
---

[TOC]

# Stochastic Processes II

Read 1.1 - 1.6 skipping 1.4

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
* The stochastic process $(N(t))_{t \geq 0}$ is called a **renewal process**. Think of replacing lightbulbs - it stays on for a certain time before being replaced. N(t) is the number of bulbs you have to replace by time t. Plotted, we see piecewise constant function, jumping to the next integer i at each $S_i$. The times at which the jumps happen are random. A Poisson process is an example of a renewal process, limited to interarrival times with exponential random variables.
* Let $F_n(t) = P(S_n \leq t)$. This function is a CDF of a sum of RVs $S_n$, may not be easy to evaluate but can still be quite useful. Observe that $N(t) \geq n \iff S_n \leq t$; the number of variables up to the time t iff the nth customer has arrived by time t. Therefore $P(N(t) =n) = P(N(t) \geq n) - P(N(t) \geq n + 1) = P(S_n \leq t ) - P(S_{n+1} \leq t) = F_n(t) - F_{n+1}(t)$. If we know the CDF then we can calculate the probabilities for N(t).
* **Renewal function**: expected number of variables up to time t. The renewal function $m: [o, \infty]\rightarrow \R$ is defined as $m(t) = E(N(t) )=$ expected number of arrivals by time t.
* Generally we observe the renewal process for a while, observe the distribution of interarrival times then try to say something about what we can expect in the future.
* <u>**Theorem**</u>: $m(t) = \sum_{n=1}^\infty F_n(t)$ and $m(t) < \infty$ for all t.
  * We can get F from the interarrival times, then calculate m and get the expected number of arrivals in some interval i in the future. m(t) is finite no matter what distribution we use. 
  * **Lemma 1**: Let X be a non-negative integer valued RV with $E(X) < \infty$ then $E(X) = \sum_{n=1}^\infty P(X \geq n)$
    * Proof: $E(X) = P(X = 1) + 2P(X=2) + 3P(X=3)$. Regrouping $=[P(X=1) + P(X=2) + ...] + [P(X=2) + P(X=3) + ...]+[P(X=3) + P(X=4) + ...] + ...$ This sum is equal to $\sum_{n=1}^\infty P(X \geq n)$. When summing non-negative quantities, we can rearrange the sum in any way and stil get the same answer.
  * **Lemma 2** (Markov's Inequality): Let X be a non negative RV with $E(X) < \infty$ then for any $t > 0,\; P(X\geq t) \leq \frac{E(X)}{t}$
    * The LHS of the inequality is often much more complicated that the RHS, so this comes in handy
    * Proof: Let y be the following RV: $y = \begin{cases} 1 & if \;x \geq t \\ 0 & else \end{cases}$. Let Z = X/t. Then if X < t we have $y = 0 \leq Z$ since Z is a nonnegative RV. If $X \geq t$ then $y = 1 \leq x / t = Z$. So y is always $\leq Z \implies E(y) \leq E(Z)$. Finally, $E(y) = 0P(y = 0) + 1P(y=1) = P(X \geq t)$ and $E(Z) = \frac{E(X)}{t}$
  * **Lemma 3**: Let X be a non-negative RV st $P(X = 0) < 1$, then $E(e^{-X}) < 1$.
  	*  Note if X were always 0 then $e^{-X}$ would always be 1. 
    * Proof: Suppose $E(e^{-X}) = 1$ then $E(1-e^{-X}) = 0$. Since X is nonnegative then $1 - e^{-X} \geq 0$. So  $1 - e^{-X}$ is a nonnegative RV whose expectation is 0. So by Markov's Inequality, $P(1 - e^{-X} \geq t) \leq 0$ for any $t > 0 \implies P(1 - e^{-X} =0) = 1\implies P(X = 0)=1$. So we get a contradiction. Thus $E(e^{-X}) \neq 1$. But $e^{-X} < 1$ always so $E(e^{-X})$ cannot be > 1 thus  $E(e^{-X}) < 1$
  * **Lemma 4**: $m(t) < \infty$ for any t
    * Proof: $P(N(t) \geq n) = P(S_n \leq t)$ by above. $= P(e^{-S_n} \geq e^{-t}) \leq \frac{E(e^{-S_n})}{e^{-t}}$. Using the fact that X's are iid, then $ = e^t (E(e^{-X_1})^N) = e^tp^n$. By Lemma 1 $p = E(e^{X_1})$. Then $m(t) = E(N(t)) = \sum_{n=1}^\infty P(N(t) \geq n) = \sum_{n=1}^\infty P(S_n \leq t = \sum_{n=1}^\infty) F_n(t)$. By the above $\sum_{n=1}^\infty P(N(t) \geq n) \leq e^t \sum_{n=1}^\infty p^n < \infty$ since $0 \leq p < 1$.
* Recall: $\mu = E(X_1) = E(X_2)=...$ which could be infinite.
* **<u>Theorem</u>**: With probability 1, $\underset{t\rightarrow \infty}{lim} \frac{N(t)}{t} = \frac{1}{\mu}$
  * We will use the Strong Law of Large Numbers (SLLN) which says that $\frac{S_n}{n} \rightarrow \mu$ with probability 1. Means that the $P(\underset{t\rightarrow \infty}{lim} \frac{S_n}{n} = \mu) = 1$. This holds for mu finite or infinite.
  * **Lemma**: $\underset{t\rightarrow \infty}{lim} N(t) = \infty$ with probability 1.
  	* As we take larger times, the number of arrivals goes to infinity.
  	* Proof: The limit always exists because N is an increasing process. $\underset{t\rightarrow \infty}{lim} N(t) < \infty \iff X_n = \infty$  for some n. This is a integer valued process, so finite if it stops at some point. $P(X_n = \infty) = 0$ so $P(\underset{t\rightarrow \infty}{lim} N(t) < \infty) = p(\cup_{n=1}^\infty \{X_n = \infty\}) \leq \sum_{n=1}^\infty P(X_n = \infty) = 0$ (Probability of union is bounded by sum of probabilities). Fact: for any events $A_1,A_2,...\; P(\cup_{n=1}^\infty A_i )\leq \sum_{n=1}^\infty P(A_i)$
  * Proof: Note $S_{N(t)} \leq t < S_{N(t) + 1} $. The customer we count up to for N(t) arrives by t but the next customer must be after t by the definition of N(t). Recall $S_n$ is the arrival time of the nth client, N(t) = # of arrivals by time t. This implies $\frac{S_{N(t)}}{N(t) } \leq \frac{t}{N(t)} < \frac{S_{N(t) + 1}}{N(t)}$. We know: $N(t) \rightarrow \infty,\; \frac{S_n}{n} \rightarrow \mu$ as $n \rightarrow \infty$. This implies $\frac{S_{N(t)}}{N(t)} \rightarrow \mu$ as $t \rightarrow \infty$. This all happens with probability 1.
  * Then $\frac{S_{N(t)}}{N(t)} = \frac{N(t) + 1}{N(t)}\frac{S_{N(t)}}{N(t) + 1} \rightarrow 1\times \mu $ as $t \rightarrow \infty$. Therefore $\frac{t}{N(t)} \rightarrow \mu$ as $t \rightarrow \infty$


