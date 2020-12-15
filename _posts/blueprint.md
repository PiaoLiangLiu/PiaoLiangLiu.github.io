> This post aims at providing a digest of Professor Xing's report "A Blueprint of Standardized and Composable ML" in BAAI, Aug 6, 2020.

# 1. Context

With the sophistication of Neural Networks in the field of machine learning, the theories and the applications are going in a divergent way. In the recent days, some researchers focus on the theory part while others are on the opposite side. To make things worse, some of them even consider it unnecessary to head in the other way.

Additionally, with the development of Neural Networks so far, the applications of them to different tasks are associated with different data - researchers design a specific algorithm and test it on a series of benchmarks with respect to a single problem - where a single model deals with all in this task. Also, the tasks feature a variety of behaviors, different levels of complexity and various cost forms. 

Therefore, Prof Xing considers it necessary to **disclose the regularity and unity behind the models.**

In contrast to how the modern ML/AI models behave, the ways in which we human deal with the problems or tasks are highly integrated. Generally, we don't think about a particular type of data(eg. Images, Texts) or do them separately. Regarding this difference, the situation faced with ML is unprecedented hard. To be more specific, in face of  a series of ML/AI algorithms, the engineers/researchers need to make a decision on choosing an algorithm to solve a problem. Neural networks are efficient and efficacious while most of them lack interpretability. On the contrary, considering the good analytical behaviors, some may prefer the classical models such as SVMs. That leads to the formation of the *zoo of algorithms and heuristics.* For example, reinforcement learning(RL) are designed for the agent to interact with the world under the supervision of reward functions. Sophisticated as it is, which forms an individual research field itself, RL is still inspired only by a particular type of problem. While it is possible for RL algorithms to deal with the tasks whose data are in the form of text, clearly there's nobody doing it in that way due to its limitations.

Professor Xing makes a metaphor that the zoo of algorithms is an airport with different runways for every different type of aircrafts. They are delicate pieces of arts but integrity is what we really desire. Therefore, targeting to get the past work more organized and expose new work, the idea of **creating a blueprint** in ML like what Maxwell did in the filed of Electricity and Magnetism emerges. 

# 2. A Blueprint of ML

Prof. Xing decomposes the blueprint of ML into three parts which are the major ingredients of today's ML studies: **Loss, Optimization solver and Architecture**.

## 2.1  Loss

It is concluded that generic standard loss takes the form of:
$$
\min_{\mathbf{q, \theta}} \mathbb{E-D-H}
$$
Where the components are:

1. $$\mathbb{E}$$ representing ***Experience*** associated with arbitrary type of data
2. $$\mathbb{D}$$ representing ***Divergence*** usually acts as a vehicle to start learning process 
3. $$\mathbb{H}$$ representing ***Uncertainty*** is a self-regularizer that control the complexity of the learned model.

Before we really step into the norm, let's take a look at few examples.

### Maximum Likelihood Estimation(MLE)

As we all know, MLE is the most classical learning algorithms taking two forms of  *Supervised* and *Unsupervised*

+ Supervised MLE
  + We have observed data $$\mathcal{D} = \{(\mathbf{x^*, y^*})\}$$
  + The objective is $$min_\theta -\mathbb{E}_{(\mathbf{x^*, y^*})\sim \mathcal{D}}[logp_\theta(\mathbf{y^*\vert x^*})]$$
  + We optimize it via SGD
+ Unsupervised MLE
  + We observe data $$\mathcal{D} = \{(\mathbf{x^*})\}$$ where $$\mathbf{y}$$ is latent variable
  + We would like to get the posterior of $$p_\theta(\mathbf{y|x})$$
  + The objective is $$min_\theta -\mathbb{E}_{\mathbf{x^*}\sim \mathcal{D}}[log\int_\mathbf{y} p_\theta(\mathbf{y^*\vert x^*})]$$
  + We solve it using EM
    + E: Imputes latent variable $$\mathbf{y}$$ through expectation on complete likelihood
    + M: supervised MLE

One of the most common interpretations of MLE is to consider it as Entropy Maximization

Here I provide the proof. There are a lot of heavy algebras so feel free to skip it.

***Proof***

Let's first formulate the problem of Entropy Maximization:
$$
\begin{aligned}
\max_{p \in \mathcal{P({X})}}& H(p)
\\s.t.\;\;
\mathbb{E}_{\mathbf{x}\sim p}[f_i(\mathbf{x})] &= \alpha_i,\;i \in\{1,2,,...,n\}
\\\mathbb{E}_{\mathbf{x}\sim p}[g_j(\mathbf{x})] &\le \beta_j, \;j \in \{1,2,,...,m\}
\end{aligned}
$$
Where $$\mathcal{P(X)}$$ is the set of probability densities on a sample space $$\mathcal{X}$$

What the problem is saying is that if we are given prior knowledge or estimates of some properties of a distribution, then subject to these constraints, the distribution is expected to be the maximum entropy distribution with the properties

Next, we conclude: ***The density*** $$p^* \in \mathcal{P(X)}$$ ***solving the optimization problem is in the exponential family***
$$
\mathcal{E(X)} =\left \{p:\mathcal{X}\rightarrow \mathbb{R}^+: p(x)=exp\left (-1-\lambda_0-\sum_{i=1}^n\lambda_if_i(x)-\sum_{j=1}^m \lambda_{n+j}g_j(x)\right) , \forall x \in \mathbb{R}\right \}
$$
Here is the proof:

Since it is a convex problem, by slater's condition, strong duality establishes. 

Then we formulate it in Lagrangian:
$$
\begin{aligned}
\mathcal{L}(p, \lambda) = -H(p)+\lambda_0(\int_\mathcal{X}p(x)dx-1)+\sum_{i=1}^n\lambda_i(\int_\mathcal{X}p(x)f_i(x)dx-\alpha_i)+\sum_{j=1}^m\lambda_{n+j}(\int_\mathcal{X}p(x)g_j(x)dx-\beta_j)
\end{aligned}
$$
Where $$\lambda_{n+j}\ge0$$

By stationarity condition of KKT:
$$
1+log(p^*(x))+\lambda_0+\sum_{i=1}^n\lambda_if_i(x)+\sum_{j=1}^m\lambda_{n+j}g_j(x) = 0
$$
Solving for $$p^*(x)$$, we have the form of exponential family.

For the $$p^*(x)$$ that satisfy the constraints, $$\alpha_i=\int_\mathcal{X}p^*(x)f_i(x)dx,\;\beta_j = \int_\mathcal{X}p^*(x)g_j(x)dx$$

After that we show that:
$$
\forall p^*(x)\in \mathcal{E},p(x)\in \mathcal{P(X)}\;,\;\;H(p)\le H(p^*)
\\ 
\begin{aligned}
H(p) &= -\int_{\mathcal{X}}p(x)log\left(\frac{p(x)}{p^*(x)}p^*(x)\right)dx
\\&= -\mathcal{D(p||p^*)-\int_{\mathcal{X}}p(x)\left(log(p^*(x))\right)dx}
\\& \le \int_\mathcal{X}p(x)\left(1+\lambda_0+\sum_{i=1}^n\lambda_if_i(x)+\sum_{j=1}^m\lambda_{n+j}g_j(x)\right)dx
\\ &\le\int_\mathcal{X}p(x)\left(1+\lambda_0+\sum_{i=1}^n\lambda_i\alpha_i+\sum_{j=1}^m\lambda_{n+j}\beta_j\right)dx
\\&=\left(1+\lambda_0+\sum_{i=1}^n\lambda_i\alpha_i+\sum_{j=1}^m\lambda_{n+j}\beta_j\right)
\\ &=\int_\mathcal{X}p^*(x)\left(1+\lambda_0+\sum_{i=1}^n\lambda_i\alpha_i+\sum_{j=1}^m\lambda_{n+j}\beta_j\right)dx
\\&= \int_\mathcal{X}p^*(x)log(p^*(x))dx
\\&= H(p^*)
\end{aligned}
$$
