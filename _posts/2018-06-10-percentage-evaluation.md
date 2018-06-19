---
layout: post
title: Tutorial - How to evaluate percentages?
---

**Note**: this is the blog version of the tutorial. If you want to reproduce the experiments, you can check the corresponding [Jupyter Notebook](https://github.com/rguigoures/tutorials/blob/master/ProportionsEvaluation.ipynb). 

Percentage is one of the most common mathematical concept. At this time of world cup, a poll has been conducted to evaluate the football enthusiasm over the french population. It appeared that 64% of the surveyed people declared they plan to watch the games.

{% include image.html url="https://rguigoures.github.io/images/french_rooster.jpg" width=350 %}

This percentage has an error estimation attached to it and the bigger the error the less likely the percentage to be accurate. This error rate is seldom mentioned in the news. The goal of this tutorial is to show how to assess whether a percentage or a probability has been correctly estimated and how big the error estimation is. Let's use the example of the poll on football as an illustration.

_Allez les bleus !!!_

#### Definition
The percentage of people planning to watch the football games is defined as the number of people planning to watch the games divided by the number of surveyed people.

<div style="background-color:#eff5fb;padding:15px;"> <font face="Monaco" size="2" color="#75787a">If 640 persons plan to watch the football games over 1000 surveyed persons, then the percentage is 64.0%</font></div>


#### Problem
What if only 10 persons would have been surveyed? Would you feel confident generalizing to the rest of the population of the country?
To answer that question, several statistical tools can be used.

## Bootstrapping

#### Definition

Bootstrapping consists in randomly sampling observations with replacement, that is, every surveyed person is independent from each other and the probability of getting an certain answer does not affect the probability of the answer of the next person. 

#### Illustration

Imagine 10 persons to be surveyed are in a room. There are also 10 interviewers. The first investigator picks someone in the room, conducts the survey and brings the person back to the room. Then, the second investigator repeats the same process. And so on with the next 8 interviewers.

This will give us the percentage of people planning to watch the games.

**Question**: is this percentage accurate?

Not sure. So we will repeat the process several times to check whether it is reliable or not.

<div style="background-color:#eff5fb;padding:15px;"> <font face="Monaco" size="2" color="#75787a">
70.0% of surveyed persons plan to watch the games.    
50.0% of surveyed persons plan to watch the games.  
60.0% of surveyed persons plan to watch the games.  
</font></div>

We can observe that for a sample of only 10 persons, the percentage is very unreliable. The challenge is to find out how many persons I need to survey to get a reliable percentage estimation.

Let's repeat the experience 1000 times for different sample sizes. The Figure below shows that 

{% include image.html url="https://rguigoures.github.io/images/bootstrap.png" width=500 %}

**Question** how can we assess the certainty of a calculated percentage? 

One option consists in computing the average value of the percentages over each trial, and the standard deviation. In the following plot, bars are for 2 standard deviation since we want to get confidence intervals with a 95% precision.

{% include image.html url="https://rguigoures.github.io/images/bootstrap_std.png" width=500 %}

<div style="background-color:#eff5fb;padding:15px;"> <font face="Monaco" size="2" color="#75787a">
For 2 surveyed persons, the percentage of persons watching the games is 40.0 (± 60.0)%  
For 4 surveyed persons, the percentage of persons watching the games is 47.5 (± 56.8)%  
For 8 surveyed persons, the percentage of persons watching the games is 52.5 (± 36.7)%  
For 16 surveyed persons, the percentage of persons watching the games is 56.3 (± 26.8)%  
For 32 surveyed persons, the percentage of persons watching the games is 62.8 (± 20.6)%  
For 64 surveyed persons, the percentage of persons watching the games is 63.4 (± 9.6)%  
For 128 surveyed persons, the percentage of persons watching the games is 64.0 (± 6.0)%  
For 256 surveyed persons, the percentage of persons watching the games is 61.1 (± 4.9)%  
For 512 surveyed persons, the percentage of persons watching the games is 63.5 (± 2.2)%  
For 1024 surveyed persons, the percentage of persons watching the games is 63.0 (± 3.7)%  
For 2048 surveyed persons, the percentage of persons watching the games is 63.7 (± 1.9)%  
For 4096 surveyed persons, the percentage of persons watching the games is 64.0 (± 1.3)%  
For 8192 surveyed persons, the percentage of persons watching the games is 64.0 (± 0.5)%  
For 16384 surveyed persons, the percentage of persons watching the games is 64.0 (± 0.7)%  
For 32768 surveyed persons, the percentage of persons watching the games is 64.1 (± 0.5)%  
</font></div>

### The binomial confidence interval

Bootstrapping is nice but sampling is a pretty heavy process for evaluating a percentage.

**Question**: Is there a way to compute directly the confidence using the numbers of surveyed persons and the percentage?

Of course, there is! This is called the binomial estimator. 

The binomial estimator considers the problem the opposite way: how likely it is that my observations are sampled from my percentage. Concretely, I know that I 64% of french people plan to watch the games, how likely is it that in my sample of 100 persons, 64 of them are going to watch the football games?

<div style="background-color:#eff5fb;padding:15px;"> <font face="Monaco" size="2" color="#75787a">
For 100 surveyed persons, the probability that 64 of them plan to watch the game, if the expected percentage is 64%, is equal to 8.29%.  
For 10,000 surveyed persons, the probability that 6,400 of them plan to watch the game, if the expected percentage is 64%, is equal to 0.83%
</font></div>

This is quite counterintuitive: the more there are observations, the less it is reliable.

But it is also more likely to find randomly 64 persons planning to watch the games over 100 surveyed people than 6,400 over 10,000.


<div style="background-color:#eff5fb;padding:15px;"> <font face="Monaco" size="2" color="#75787a">
The probability to randomly find 64 persons planning to watch the games over 100 surveyed people is 1.0%.  
The probability to randomly find 6,400 persons planning to watch the games over 10,000 surveyed people is 0.01%.
</font></div>

**Question**: How to derive the confidence interval from it?

Let's do some maths! Let's write the binomial estimator as a conditional probability: \\(P(k \mid n, p)\\). 

**Example**: In the survey published in the news, 1,000 persons were surveyed. Then, \\(P(k=640 \mid n=1000, p=0.64)\\) denotes the probability to find 640 persons planning to watch the games over 1,000 surveyed people knowing that the expected percentage is 64%.

Let's suppose that we want to have a confidence interval with a 95% precision, that is an error rate of 5%.

We can then iterate over k from 1 and sum \\(P(k \mid n, p)\\), until we hit half of the error rate, i.e 2.5%:

$$
\begin{align}
&P(k=1 \mid n=1000, p=0.64)\\
+&P(k=2 \mid n=1000, p=0.64)\\
+&... \\
+&P(k=x_L \mid n=1000, p=0.64) = 2.5\%
\end{align}
$$


<div style="background-color:#eff5fb;padding:15px;"> <font face="Monaco" size="2" color="#75787a">
Lower bound of the 95% confidence interval is when we hit k=610
</font></div>

Now we have found the lower bound of the confidence interval, let's do the same for computing the upper bound:

$$
\begin{align}
&P(k=1000 \mid n=1000, p=0.64)\\
+&P(k=999 \mid n=1000, p=0.64)\\
+&... \\
+&P(k=x_U \mid n=1000, p=0.64) = 2.5\%
\end{align}
$$


<div style="background-color:#eff5fb;padding:15px;"> <font face="Monaco" size="2" color="#75787a">
Upper bound of the 95% confidence interval is when we hit k=670
</font></div>

**Conclusion**: For 1,000 surveyed persons, the percentage to find 640 persons pretending planning watching the games  is equal to 64 (± 3)%, with a 95% precision. 

**Question**: Is there a way to compute it directly?

Yes. There is also a way to estimate that value:

$$e = z \displaystyle\sqrt{\dfrac{p(1-p)}{n}}$$

where \\(p\\) is the percentage, \\(n\\) is the number of samples and z is the z score, constant value depending on the required precision.


<div style="background-color:#eff5fb;padding:15px;"> <font face="Monaco" size="2" color="#75787a">
Error rate is: 9.41
</font></div>

Now let's plot the same chart as we did for bootstrapping. And note that we obtain similar confidence intervals.

{% include image.html url="https://rguigoures.github.io/images/binomial_estimator.png" width=500 %}

<div style="background-color:#eff5fb;padding:15px;"> <font face="Monaco" size="2" color="#75787a">
For 2 surveyed persons, the percentage of persons watching the games is 64.0 (± 66.5)%  
For 4 surveyed persons, the percentage of persons watching the games is 64.0 (± 47.0)%  
For 8 surveyed persons, the percentage of persons watching the games is 64.0 (± 33.3)%  
For 16 surveyed persons, the percentage of persons watching the games is 64.0 (± 23.5)%  
For 32 surveyed persons, the percentage of persons watching the games is 64.0 (± 16.6)%  
For 64 surveyed persons, the percentage of persons watching the games is 64.0 (± 11.8)%  
For 128 surveyed persons, the percentage of persons watching the games is 64.0 (± 8.3)%  
For 256 surveyed persons, the percentage of persons watching the games is 64.0 (± 5.9)%  
For 512 surveyed persons, the percentage of persons watching the games is 64.0 (± 4.2)%  
For 1024 surveyed persons, the percentage of persons watching the games is 64.0 (± 2.9)%  
For 2048 surveyed persons, the percentage of persons watching the games is 64.0 (± 2.1)%  
For 4096 surveyed persons, the percentage of persons watching the games is 64.0 (± 1.5)%  
For 8192 surveyed persons, the percentage of persons watching the games is 64.0 (± 1.0)%  
For 16384 surveyed persons, the percentage of persons watching the games is 64.0 (± 0.7)%  
For 32768 surveyed persons, the percentage of persons watching the games is 64.0 (± 0.5)%  
</font></div>

### Comparing two proportions

Imagine you want to compare the evoluation of a percentage month over month. Let us use another example here. The news claim that satisfaction rate of citizens with the actions lead by the president dropped from 45% to 43%. A survey has been conducted on a sample of 1,000 persons.

**Question**: How can I assess whether this change is relevant or not?

The first solution would be to compare the confidence intervals of the two proportions.


<div style="background-color:#eff5fb;padding:15px;"> <font face="Monaco" size="2" color="#75787a">
The satisfaction rate in May is 45.0 (±3.0)%  
The satisfaction rate in June is 43.0 (±3.0)%  
</font></div>

In this case, we observe a decrease of 3pp of the satisfaction rate between May and June. The confidence intervals overlap but this is not enough to draw any conclusion.

**Question**: Can we directly compute a confidence interval on the decrease itself?

Yes, in a similar way to how we do it for the proportions:

$$e = z \displaystyle\sqrt{\dfrac{p_1(1-p_1)}{n_1} + \dfrac{p_2(1-p_2)}{n_2}}$$

where \\(p_1\\) and \\(p_2\\) are the percentages and \\(n_1\\) and \\(n_2\\) the number of surveyed persons, of respectively the first and the second month.


<div style="background-color:#eff5fb;padding:15px;"> <font face="Monaco" size="2" color="#75787a">
We observe between May and June a decrease of 2.0 (±4.0) pp of the satisfaction rate
</font></div>

This means that the sample size of the surveyed population is too small to generalize to the rest of the country.
