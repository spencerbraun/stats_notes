[TOC]

# Data Driven Impact

* Learning objectives for technical members: learning the real world limitations and contexts for our models, seeing how learning methods are deployed in real companies, going beyond learning the methods from classes and understanding what is effective. How do we choose a learning method given limitations or existing data.
* Adaptive experiments, bandits, reinforcement learning the frontier replacing A/B testing. Adaptive experiments where the probability you receive certain emails can update every day or week. Creates faster learning

## Class Topics

### Machine Learning

* In slides, age is a confounder for death and treatment - correlated to both. Once included treatment effect increases and becomes signficant. 
* Treatment is for specific disease, so including disease is adding a highly correlated covariate. Interpreting the individual coefficients becomes more difficult, while prediction may not be harmed.

## The Power of Experiments

### Chapter 4: Experimentation Considerations

* Barriers to experimentation
  * Not enough participants - achieving statistical power for certain kinds of randomization can be difficult - consider the level of randomization needed to understand if power can be achieved
  * Randomization can be hard to implement - communications are an easy touchpoint for experimentation, along with tech / digital infrastructure 
  * Experiments require data to measure their impact - choosing the right metric is challenging, the easiest to measure may skew incentives. Start with data audit to understand what data you already have, then consider additional data needed for desired inferences.
  * Underappreciation of decision makers’ unpredictability - people sometimes make weird and unpredictable decisions.
  * Overconfidence in our ability to guess the effect of an intervention 
* Log experiments centrally so teams have access to past experiments and don't have to start from scratch
* Build infrastructure so non-statisticians can run successful experiments, with a centralized expert team designing the platform and providing specialized assistance.

## Running Randomized Evaluations

### Chapter 3: Asking the Right Questions

* We need to have good descriptive information on the participants and the context of the program, know the objectives of the program, and have good process indicators that will tell us whether the program we are evaluating was well implemented.
* Start with strategic questions - what are the goals, make sure they are well-defined.
* Then turn to descriptive questions - what is the reality on the ground, what are the problems faced.
* Process questions - how well is the program being implemented? Are things actually running smoothly? If current program is not carried out correctly, may have higher impact correcting current policies than trying new ones.
* Impact questions - did it work?
* A needs assessment is a collection of qualitative and quantitative information meant to provide descriptive info on the context of the situation. The basis for a program design, helps us find weaknesses in current approaches, develop methodology for impact assessment. May be entirely sufficient if there is no problem, the problems are not prioritized by participants.
* Lit review may reveal we don't need to experiment - already have promising results we could implement.
* Business case assessment - how cost effective are the interventions, considering your context. In your best-case scenario, would your findings be a cost-effective solution that could be rolled out? Requires assumptions about costs and impact, scenario analysis is helpful.
* Impact evaluations can help answer questions like which alternative to pursue, which elements of a program matter most, can we work on multiple problems concurrently.
* Eventually need to choose which questions to address with randomized evaluation. Consider factors like the potential influence of the information yielded by evaluation. 
* When the question we want to answer is crosscutting, we must design our evaluation carefully so that we are isolating just the one factor we want to examine - orthogonalize.

