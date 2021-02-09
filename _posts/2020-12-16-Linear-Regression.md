---
layout:     post   				    # 使用的布局（不需要改）
title:      Linear Regression 		# 标题
date:       2020-12-15 				# 时间
author:     Shuai Liu 						# 作者
header-img: img/post-ml.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - machine learning
    - data science
    - CS 189


---

# Linear Regression

## 1. Introduction

### Some terminologies

**Target** $$y \in \mathbb{R}$$ something we'd like to predict

**Observations** $$\mathbf{x} \in \mathbb{R}^k$$: summary of what we know about $$\mathbf{y}$$

**Regression**: the true relationship between $$\mathbf{x}$$ and $$y$$  takes the form of $$y=f(\mathbf{x})$$

**Hypothesis**: our approximation of the true relationship between $$\mathbf{x}$$ and $$\mathbf{y}$$

**Hypothesis Class**: solutions to hypothesis are restricted to hypothesis classes by imposing some assumptions

**parametric model**: there is a finite-dimensional vector $$\mathbf{w} \in \mathbb{R}^d$$ determines the behavior of the model.

**Loss Function**: to measure the performance of our models



## 2. Ordinary Least Squares(OLS)

OLS is one of the simplest problems in regression tasks. 

Here is the basic background of linear regression: With the basic assumption that the *relationship* between $$y \text{ and } x$$ is a linear one, that is we assume: $${y_i=w^Tx_i}$$, writing in matrix takes the form:
$$
\mathbf{y} = Xw \tag{1}\\
\begin{equation*}
\mathbf{y} = 
\begin{pmatrix}
y_{1}  \\
y_2 \\
\vdots\\
y_n 
\end{pmatrix}
\;\;
X = 
\begin{pmatrix}
x_1^T \\
x_2^T \\
\vdots \\
x_n^T
\end{pmatrix}
\end{equation*}
$$
$\mathbf{x} \text{here is our experiment}$

Here $\mathbf{y} \in \mathbb{R}^n$ is a *column vector* where each element is the $y_i$ previously and each row of $X$ is our $x_i$. Sometimes, $X \in \mathbb{R}^{n \times d}$ is also called *design matrix*. Please note that the convention we use here is usually the implicit assumption and it is redundant to mention it every time. So we can come up with the *loss function* of the problem: $\mathcal{L} = \frac{1}{n}\sum_{i=1}^n(y_i-w^Tx_i)^2$. It is also called *MSE(Mean Squared Error)* in machine learning. Writing the loss function in a matrix form, we can get:
$$
\mathcal{L} = \frac{1}{n}\|\mathbf{y}-Xw\|_2^2  \tag{2}
$$
Since the loss measures how bad our model approximates the real data, our goal is to minimize $\mathcal{L}$ in (2).All the remaining is how to solve (2). We introduce 2 ways in the following sections.



## 3. Vector Calculus

The first way we introduce is using vector calculus to find the optimal solution to (2). I'll briefly introduce vector calculus in Appendix at the end of this post. But first let's get into the loss function together!

I'd like first to clear that $\mathcal{L}$ in (2) is a function about $w$ since all of the others in (2) are given data. Also, I'd like to introduce two vector calculus formulas that we will use later. If you'd like to know the proof,  see the appendix! 
$$
x,w \in \mathbb{R}^k, M \in \mathbb{R}^{k \times k}\\
\begin{equation*}
\begin{aligned}

\nabla_x x^TMx &= 2Mx \\
\nabla_x w^Tx &= w
\end{aligned}
\end{equation*}
$$
Then we could easily find out that $\mathcal{L}$ is a *convex function* of $w$. Therefore, the optimal solution $w^*$ is the one that where the gradient of $\mathcal{L}$ w.r.t. $w$ is 0:
$$
\begin{aligned}
\arg \min_w \mathcal{L}&=\arg\min_w \|\mathbf{y}-Xw\|_2^2\\
&=(\mathbf{y}-Xw)^T(\mathbf{y}-Xw)\\
&= w^TX^TXw - 2\mathbf{y}^TXw+\mathbf{y}^T\mathbf{y}\\ \\

\end{aligned}
\tag{3}
$$

$$
\nabla_w\|\mathbf{y}-Xw\|_2^2=2X^TXw - 2X^T\mathbf{y} \tag{4}
$$

The vector calculus formulas provided above is used in (4). Assume $X^TX$ is invertible, we can get our final solution:
$$
\begin{aligned}
\nabla_{w^*}\|\mathbf{y}-Xw\|_2^2 &= 0 \\
2X^TXw^* - 2X^T\mathbf{y} &= 0 \\
\end{aligned}
\\
$$

$$
w^* = (X^TX)^{-1}X^T\mathbf{y} \tag{5}
$$


## 4. Orthogonal Projection

Another approach for solving (2) is a linear algebraic one. Before we really step into this approach, some theorems of linear algebra must be first understood.

Assume $S ,S^\perp \subset \mathbb{R}^m$ where each of them are orthogonal complement of each other. Consider arbitrary vector $v \in \mathbb{R}^m$, we have 
$$
v= v_S + v_{S^\perp}, v_S \in S,  v_{S^\perp} \in S^{\perp}
$$
Let $P_sv$ be the project of $v$ to $S$, then
$$
\|v-P_sv\|_2 \le\|v-s\|_2,\; \forall s \in S 
\tag{6}
$$
The inequality is tight if and only if $s=P_sv$

*Proof*:
$$
\begin{aligned}
\|v-s\| &= \|v-P_sv+P_sv-s\|
\\&=\|v-Ps_v\|+\|P_sv-s\|
\\&\ge \|v-Ps_v\|
\end{aligned}
$$
The last two step uses Pythagorean theorem since $v-P_sv \in S^{\perp}, P_sv-s\in S$ 

Also, knowing Fundamental Theorem of Linear Algebra([FTLA](https://mathworld.wolfram.com/FundamentalTheoremofLinearAlgebra.html)) is necessary: 

Here we'll only use 
$$
range(X)^\perp = null(X^T)
$$
Back to our problem: (2), $Xw\in range(X)$ and we could view it as a vector in the space spanned by the column vectors of $X$ (column space). From (6) we could know that the minimum of $\|\mathbf{y}-Xw\|$ is achieved when $Xw=P_X\mathbf{y}$. Also, by FTLA, we have 
$$
\mathbf{y}-Xw^* = \mathbf{y}-P_X\mathbf{y} \\
\mathbf{y}-P_X\mathbf{y}\in range(X)^\perp \\
range(X)^\perp= null(X^T)\\
\mathbf{y}-Xw^* \in null(X^T)
$$
Finally, by the definition of null space, we have
$$
X^T(\mathbf{y}-Xw^*) = \mathbf{0}
$$
Here we could achieve exactly the same solution as (5)

What's worth mentioning is that if $X$ is a full rank matrix, then $Xw$ should be unique since the column vectors of $X$ is linearly independent from each other. The solution to the problem (2) is thus unique under this condition. However, if $X$ is not full rank, the solution may be multiple and we'll deal with it later.



## 5. Ridge Regression



## 6. Appendix: Vector Calculus

 