---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Bridging the Gaps With Reinforcement Learning"
subtitle: ""
summary: ""
authors: ["admin"]
tags: ["Reinforcement Learning", "Deep Learning"]
categories: ["Reinforcement Learning", "Deep Learning"]
date: 2019-11-04T03:10:09+05:30
lastmod: 2019-11-04T03:10:09+05:30
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: "Center"
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
In this post, I will be talking about a unique way to use reinforcement learning (RL) in deep learning applications. I definitely recommend brushing up some deep learning fundamentals and if possible, some policy gradient fundamentals as well before you get started with this post.

Traditionally, RL is used to solve sequential decision making problems in the video game space or robotics space or any other space where there is a concrete RL task at hand. There are several applications in the fields of video games and robotics where the task at hand can be very easily seen as an RL problem and can be modeled appropriately as well. But RL as a technique is quite versatile and can be used in several other domains to train neural networks that are traditionally trained in a supervised fashion. We'll talk about one very important such application in this post. Along the way, I'll also try to convince you that this isn't really a different way to use RL but rather just a different way to look at the traditional RL problem. So with that, lets begin!

## Non-differentiable steps in deep learning: The Gaps

Sometimes when we're coming up with neural network architectures, we may need to incorporate some non-differentiable operations as a part of our network. Now this is a problem as we can't backpropagate losses through such operation and hence lets call these "gaps". So what are some common gaps we come across in neural networks?

On a side-note, before we start talking about some "real gaps", its worth mentioning that the famous ReLU function is a non-differentiable function but we overcome that gap by setting the derivative at 0 to either 1 or 0 and get away with it. 

Now lets take a better example -- variational autoencoders (VAE). Without going into two many details, the VAE network outputs two vectors: a $\mu$ vector and a $\sigma$ vector and it involves a crucial sampling step where we sample from the distribution _N($\mu$, $\sigma$)_ as a part of the network. Now sampling is a gap as it is a non-differentiable step. You should stop here and convince yourself that this is, in fact true. When we sample, we don't know what the outcome will be and hence the function in not differentiable. So how do they get over this in the VAE case? They use a clever trick.
Instead of sampling from _N_($\mu$, $\sigma$), they just rewrite this as $\mu$ + $\sigma$_N(0,1)_ where they sample from the standard normal function. This neat trick now makes the expression differentiable because we just need the $\mu$ and $\sigma$ quantities to be differentiable and we don't care about the _N(0,1)_. Remember that we only need to differentiate with respect to the parameters of our network (brush up some backpropagation basics if you're confused here) and hence we need to differentiate only with respect to $\mu$ and $\sigma$ and not the standard normal distribution. For more details about VAEs read [this post](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf) or [this one](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73).

{{< figure src="vae.PNG" title="Variational Autoencoders. Source: [here](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)" lightbox="true" >}}

So as it turned out, that wasn't a very good example either but we're starting to understand what we mean by gaps now and how common they are. Some common examples of gaps in networks are sampling operations and the argmax operation. Once again, convince yourself that argmax is not a differentiable function. Assume you have a function that takes argmax of two quantities _(x1,x2)_. When _x1_ > _x2_, this has value 0 (zero-indexed) and when _x1_ < _x2_, it has value 1. Say the function is not defined on the _x1==x2_ line or define it as you wish (0 or 1). Now if you can visualise the 2D plane, you'll see that the function is not differentiable as we move across the _x1==x2_ line. So argmax isn't differentiable _but max is a differentiable function_ (recall max pooling in CNNs). Read [this thread](https://www.reddit.com/r/MachineLearning/comments/4e2get/argmax_differentiable/) for more details.

These functions are commonly used in natural language processing (NLP) applications, information retrieval (IR) applications and Computer Vision (CV) applications as well. For example, a sampling function could be used to select words from a sentence based on a probability distribution in an NLP application or an argmax function could be used to find the highest ranked document in an IR application. [Hard attention](https://jhui.github.io/2017/03/15/Soft-and-hard-attention/) uses sampling techniques which involves non-differentiable computation.

So its quite clear that these gaps are common in several deep learning architectures and sometimes, it could even be useful to introduce such a gap in the network intentionally to reap added benefits. The only question is, _how do we bridge these gaps?_

## Reinforcement Learning and Policy Gradients: The Bridge

Policy gradients are a class of algorithms in RL. There are several policy gradient algorithms and [this](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html) is a great blog that lists out almost all of them. But without going into too many details, these algorithms work in the policy space by updating the parameters of the policy we're trying to learn. That means we don't necessarily need to find the value function of different states but we can directly alter our policy until we're happy. 
The most common policy gradient (PG) algorithm is the REINFORCE which is a Monte Carlo algorithm. This means we run an entire episode and make changes to our policy only at the end of each episode and not at every step. We make these changes based on the returns that we received by taking a given action from a given state in the episode. I skip the derivation of the policy gradient here but it can be found in the link above. The final result is in the image below.

{{< figure src="pg.PNG" title="The Policy Gradient. Source: [here](https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63)" lightbox="true" >}}

The key idea here is that in policy gradient methods, we are allowed to _sample different actions from a given state and wait till the end of an episode before we make updates to our network_. So if we have a sampling operation as a part of our network, we can introduce a policy gradient and think of it as sampling actions in a given state in an RL framework. A similar procedure can also be followed if we had argmax in place of the sampling operation.

Consider a neural network now with a gap. The images below are taken from [this blog](http://karpathy.github.io/2016/05/31/rl/) on Policy Gradients written by Andrej Karpathy.

{{< figure src="karpathy1.PNG" title="Gaps in a neural network. Source: Karpathy's [blog](http://karpathy.github.io/2016/05/31/rl/)" lightbox="true" >}}
{{< figure src="karpathy2.PNG" title="The sampling operation. Source: Karpathy's [blog](http://karpathy.github.io/2016/05/31/rl/)" lightbox="true" >}}

So now we can train the blue arrows i.e. the differentiable path as usual. But to train the red arrows, we need to introduce a policy gradient and as we sample, we ensure with the help of the policy gradient that we encourage samples that led to a lower loss. In this way, we are "training" the sampling operation or one could say, propagating the loss through the sampling operation! Note that the updates to the red arrows happen independently than those of the blue arrows.
Note that in the diagrams above, there isn't really a gap per-say because the blue arrows go all the way from start to finish. So there is a differentiable path and a non-differentiable path. A true gap would mean there would be no completely differentiable path at all. In this case, we need to make sure that the loss functions on either side of the gap are "in sync" and are being optimized in such a way that it facilitates joint training and achieves a common goal. This is often not as easy as it sounds.

I said at the start that as obscure as it seems, this is still the traditional RL problem we're used to with the MDP and states and actions. We can still look at this entire setup as a traditional RL problem if we think of the inputs to the neural network as the state and the sampling process as sampling different actions from that given state. Now what is the reward function? This depends on what comes after the gap and could be an output from the rest of the network or it could be a completely independent reward function that you came up with as well. So at the end of the day, it is still the same MDP with the traditional setup but just used in a very different way. 










