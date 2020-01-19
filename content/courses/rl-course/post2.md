---
title: "RL Fundamentals and MDPs"
linktitle: "RL Fundamentals and MDPs"
toc: true
type: docs
date: "2019-05-05T00:00:00+01:00"
draft: false
menu:
  rl-course:
    parent: The RL Course
    weight: 2

# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 2
---
In this post, we'll try to get into the real nitty-gritties of RL and build on the intuition that we gained from the [last article]({{< ref "/courses/rl-course/post1.md" >}}). So we'll bring in some mathematical foundation and then introduce some RL parlance that we will use for the rest of this course. I'll stick to the standard notation and jargon from the RL book.

## Two Issues

Before we begin with the mathematical foundations of RL, I'd like to point out some issues with RL and what kinds of problems we need to account for if we were to come up with RL algorithms of our own. Once again, I'm going to move on to a new example so lets take Chess this time.
So we want to teach our agent (recall what this means from the previous post) how to play the game of chess. Lets assume we have some sort of reward function in place where we get some small positive rewards for capturing a piece and small negative rewards for losing a piece depending on the importance of the piece (so losing a queen would lead to a negative reward of larger magnitude than losing a pawn). We also have some large positive final reward for winning the game and a large positive negative reward for losing. If you're wondering whether just ths large final reward is a sufficient reward function on its own, you're probably right and it probably is, but lets stick to this for the sake of illustration.
Now assume we have an RL algorithm that can look at several games of Chess and the rewards and learn to play Chess on its own. What would this algorithm need to account for? We spoke about trial and error being the basis of any RL algorithm in the previous post that is exactly what our algorithm would do as well. It starts playing random moves and when it plays a good move (positive reward), it remembers to play that move the next time it is in a similar situation. This seems fine on the face of it, but there is an issue. Maybe the algorithm found a good move to play at a given position, _but what if there was a better move?_
We need some way for the algorithm to account for the possibility of there being a better move than the one it has found already. So when we train our agent we need to make sure the agent doesn't greedily play the best move it knows all the time but also plays some different moves, hoping that they may be better than the one it already found. This is called the _exploration-exploitation tradeoff_ in RL. Usually, RL algorithms tend to explore i.e. play many random moves initially and when the agent is more sure about the best moves under different circumstances, it starts exploiting that knowledge. We will revisit this problem in the next post with another example.

Lets move on to the next issue that our RL algorithm will have to account for. Lets say our RL algorithm is learning from a game of Chess again where the player sacrifices the queen but goes on to win the game. The RL agent immediately registers a negative reward for the loss of the queen but the large positive reward for winning the game only comes much later. But it is entirely possible that the very queen sacrifice that the RL agent probably classified as a bad move, was the reason for the player winning the game. So it is possible that a move seems like a "bad" one in the short term but in the long term, could be a very "good" move. How do we account for this in our algorithm? This is the concept of _delayed rewards_ and we will deal with a simple yet elegant solution for this as well as we go through this post.

With that background, lets talk about how RL problems are modeled and get into some math.

## Markov Decision Processes

> The future is independent of the past, given the present

{{< figure library="true" src="post2_mdp_example.jpg" title="A Markov Decision Process. Source: [here](https://randomant.net/reinforcement-learning-concepts/)" lightbox="true" >}}
{{< figure library="true" src="post2_rl_with_not.jpg" title="The RL Framework (with some additional details). Source: Google Images" lightbox="true" >}}

Almost all RL problems can be modeled as a Markov Decision Process (MDP). So what is an MDP? An MDP is a mathematical model that we will be seeing a lot over this course. Most of our RL environments, will be MDPs. An MDP can be defined as a five tuple:
$$\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$ 

Lets take a closer look at what all of this means.

* _S_ : This is the set of states of the MDP. In an RL setting, this would correspond to different settings of the environment. In the previous post, we spoke about how RL was all about choosing the right actions at the right times i.e. depending on the state of the environment. This is the state we were talking about. A state in chess or tic-tac-toe cpuld be the board positions or while riding a bike could be some combination of the pertinent variables like the angle of the bike with thr ground, the wind speed, etc. The S variable represents the set of all unique states in the MDP. 
* _A_ : This is the set of all actions of the MDP. We already spoke about actions briefly. Actions describe the possible moves in a game of Chess or tic-tac-toe or different arrow keys or buttons in a video game, etc. It represents the different options the agent has and can play at a given point in time. A represents the set of all unique actions available to the agent.

It is worth mentioning now, that both _S_ and _A_ can be finite or infinite sets. Both can also be continuous spaces or discrete spaces, depending on the variables in the state space or the nature of actions. For example, if we have a variable taking real numbered values in the state space, the state space is automatically continuous (and infinite). If our actions are in the form of degree of turning a steering wheel, once again, continuous and infinite action space.

Before we move on to the other symbols, lets get some things clear. Here is another more detailed MDP for your reference.

{{< figure library="true" src="post2_mdp.png" title="Another MDP. Source: Google Images" lightbox="true" >}}

Some of the states of the MDP are designated as start states or initial states and end states or terminal states. An _episode_ in RL is a sequence of state-action pairs that take the agent from a start state to a terminal state. So the agent starts from one of the intial states, plays an action, goes to the next state and so on until it hits a terminal state and the episode ends. Assume for now that there is such an end i.e. every episode does end after some unknown, finite time. This is a property of finite horizon MDPs which is what we will stick to in this course.
Now lets take a look at this MDP in the diagram above. Assume _S_<sub>0</sub> is your initial state. Notice that _a_<sub>0</sub> from _S_<sub>0</sub> has two arrows, one going into _S_<sub>0</sub> again and another going into _S_<sub>2</sub>. The numbers on the arrows indicate 0.5 and 0.5 respectively. This means that if an agent plays the action _a_<sub>0</sub> from _S_<sub>0</sub>, it has a 0.5 probability that it ends up back in _S_<sub>0</sub> and a 0.5 probability that it ends up in _S_<sub>2</sub>. And similarly we have arrows going all over the diagram. Also notice the wiggly arrows -- they're rewards.
Notationally, we index the sequence of state-action pairs in an episode with a time variable _t_ so we say an agent plays action _a_<sub>t</sub> from state _s_<sub>t</sub> abnd gets reward _r_<sub>t+1</sub> for doing so (the reward can be 0). Here, _t_ starts from 0 and we represent the terminal time-step as _T_.

Another crucial thing to note is that actions in an MDP have to be instantaneous in nature. That means you take action _a_<sub>t</sub> from state _s_<sub>t</sub> and end up in state _s_<sub>t+1</sub> immediately. There are several actions in real-world problems that may not be instantaneous but we'll deal with them later. For now, assume your actions are instantaneous.

* _P_ : Now P is the probability function defined as _P_(s<sup>'</sup>| s, a) which is read as the probability of moving to "state s'" from "state s" if the agent plays "action a". So for example, _P_(_S_<sub>2</sub>|_S_<sub>0</sub>,_a_<sub>0</sub>) = 0.5
* _R_ : This is the reward function and is defined as _R_(s<sup>'</sup>| s, a) which is the reward the agent gets for moving to "state s'" from "state s" if the agent plays "action a". So for example, _R_(_S_<sub>0</sub>|_S_<sub>1</sub>,_a_<sub>0</sub>) = +5. Rewards are always scalars.
* $\gamma$ : We spoke about the concept of delayed rewards earlier in the post and we wanted a way to accound for delayed effects of actions. This is where $\gamma$ helps. We define the _returns_ of an action from a given state as the sum of the _discounted rewards_ we receive from that state for playing that action. If we started from the state _s_<sub>0</sub>, the returns would be defined as _r_<sub>1</sub> + $\gamma$ _r_<sub>2</sub> + $\gamma$<sup>2</sup> _r_</sub>3</sub> + ... $\gamma$ <sup>_T-1_</sup> _r_<sub>_T_</sub>. We use the word "discounted" because $\gamma$ is usually a number between 0 and 1 and with the increasing powers, we give more weight to the immediate rewards than the delayed rewards. Hence, $\gamma$ is also called the discounting factor. The symbol we use for returns from timestep _t_ is usually _G_<sub>_t_</sub> although some people like using _R_ as well (_r_ for reward and _R_ for returns). We will stick to the former notation. Now going back to the queen sacrifice example, if we were to consider the returns in our algorithm instead of just the immediate reward, we will be able to account for the delayed positive effect and not just the immediate negative effect.

But with all of that information, we still haven't covered that blockquote at the start of this section. No, that wasn't a quote from Avengers: Endgame and it is the most important takeaway from this post. So what does it mean. Mathematically, it means this:

$$P(s\_{t+1}|s\_t,a) = P(s\_{t+1}|s\_t,s\_{t-1},s\_{t-2},s\_{t-3},...,a)$$

But what does that mean intuitively? It means are probability of going to state _s_<sub>t+1</sub> from state _s_<sub>t</sub> under the action _a_ is independent of how we got to state _s_<sub>t</sub> in the first place. If you're thinking this isn't very realistic, you're right but this type of modeling works in most cases and is called _The Markov Property_. So we will stick to this for now.

And with that, we've covered MDPs and how to model RL problems. With this background, I also recommend reading [this]({{< ref "/post/modeling-rl-problems/index.md" >}}). But with that definition, we still haven't accounted for the exploration-exploitation tradeoff. So in the [next post]({{< ref "/courses/rl-course/post2.md" >}}), we'll introduce a few more symbols and definitions and get cracking with our very first RL algorithm!

Once again, let me know if you have any feedback or suggestions.

### References

1. [A (Long) Peek into Reinforcement Learning by Lilian Weng](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)

