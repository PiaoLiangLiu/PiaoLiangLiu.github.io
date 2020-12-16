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

Machine Learning is a huge topic in which numerous problems are included, eg. regression, classification, etc. Whatever the problem is, the basic task of machine learning is extracting a *relationship* from data. In regression tasks, the relationship takes the form of $$ \mathbf{y}=f(\mathbf{x}) $$. It could be understood as predicting some target $$\mathbf{y}$$ given some summary of the observations $$\mathbf{x}$$. We typically consider the target 

$$\mathbf{y} \in \mathbb{R}$$  as some sort of quantity that we'd like to know and $$\mathbf{x} \in \mathbb{R}^k$$ as some sort of numerical observations.  For example, having a basic idea of the house prices of some area is necessary if someone would like to purchase one in this area. Given some observations such as area, number of bedrooms, restaurants around it, etc, it is possible to estimate the price. Here comes the usage of machine learning techniques! It's really cool to have an accurate estimation given some information of the house, isn't it? In a typical machine learning problem, we have no knowledge about this true relationship. All we have is data $$ \mathcal{D}=\{x_i, y_i\}_{i=1}^n$$ and we'd like to infer or approximate this true underlying relationship by designing and performing algorithms. Back to our regression task, our final product would be $$\hat{\mathbf{y}}=h(\mathbf{x})$$ called **hypothesis** which is our approximation of $$f(\cdot)$$. Since it is intractable to learn an arbitrary function, we usually restrict our solutions to a **hypothesis class ** $$\mathcal{H}$$ by imposing some assumptions. To be more specific, we usually employ a **parametric model** meaning that there is a finite-dimensional vector $$\mathbf{w} \in \mathbb{R}^d$$ determines the behavior of the model. The elements of $$\mathbf{w}$$ is called weights or parameters. We usually use a **Loss Function** to measure the performance of our models. Instructed by the loss function, we could search the best solution to the task. We'll start with **linear regression**, one of the most classic algorithms in Machine Learning and hope you guys enjoy it.



## 2. Ordinary Least Squares(OLS)

OLS is one of the simplest problems in regression tasks. 

Here is the basic background of linear regression: With the basic assumption that the *relationship* between $$y \text{ and } x$$ is a linear one, that is we assume: $${y_i=w^Tx_i}$$, writing in matrix takes the form: