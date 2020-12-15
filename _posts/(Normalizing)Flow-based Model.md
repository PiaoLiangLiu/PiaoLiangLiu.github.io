# (Normalizing)Flow-based Model

> In this post, we are going to have a look at a type of generative model: Flow-based model. Different from other generative models like GAN or VAE, flow-based model explicitly learns the input data distribution.



As is known to all, generative modeling problem is an extremely challenging one in the world of machine learning. A promising generative model is expected to have the following traits:

1. Learning realistic world models which allows the agent to plan before interacting with the world, requiring few or no human supervision.
2. Being able to model all the dependencies within high-dimensional input

Comparing with VAE and GAN, flow-based model gains little attention in this field. The basic idea of *Flow-based Model*, especially *Normalizing Flow* is to learn a series of **invertible** transformation with which the agent could use a simple and familiar distribution(eg. Gaussian) to model the complicated and high-dimensional input data and vice versa. Facilitated with this powerful statistical approach, one could efficiently complete various downstream tasks: sample unobserved but realistic new data points (data generation), predict the rareness of future events (density estimation), infer latent variables, fill in incomplete data samples, etc.



## Math Recap

Before we  step into normalizing flow, we need to get to know two basic math tools: the determinant of Jacobian and the change of variable rule. They are very basic rules, so feel free to skip.

### Jacobian Matrix and Determinant





