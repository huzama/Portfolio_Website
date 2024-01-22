---
layout: post
title:  Methodology Section for Semester Project
date:   2023-06-15 
description: A sample of technical writing
tags: links
categories: Writing
---
<h1> Method </h1>

In this study, we propose a model that generates abstractive summaries of news articles while maintaining factual consistency with the original article. The architecture of our model is illustrated in Figure \ref{fig:method}. It consists of fine-tuning a large language model using Reinforcement Learning as described in Section \ref{ssec:design}. We leverage a pre-trained model designed for Natural Language Inference task as a reward model to guide the model to make the summaries factually consistent. The details of computing reward are explained in Section \ref{ssec:reward}.

We ran all of our experiments on a multi\_news dataset. This dataset comprises multiple news articles covering the same event, each accompanied by a summary. However, in our experiment, we treat each news article as an individual input to our model, eliminating the need for supervised settings or utilizing the provided summaries from the multi\_news dataset.


<h2> Model Design </h2>
We utilize the FLAN-T5-XXL language model, known as the summary generator, for our article summaries. The model is prompted in a traditional manner to generate a summary, which is then evaluated using a reward function that assigns a numerical value between -1 and 1. A score of -1 represents the worst reward, while a score of 1 indicates the best reward. The article-summary pair and the reward are then passed to the PPO trainer (Vonwerra, 2022), depicted in Figure \ref{fig:ppo}.

The PPO trainer operates by first calculating the KL-Divergence between the updated and original models' outputs. This KL value ensures that the trained model does not deviate significantly from the original model. If it begins to drift away, a KL penalty is applied to prevent further deviation. Subsequently, the KL value is fed into the optimizer along with the computed reward. The model then computes the loss using the following equation  \ref{eq:ppo}.

\begin{equation}
   L(s,a,\theta_k,\theta) = \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a)
    \label{eq:ppo}
\end{equation}

In this equation, &pi;	represents the policy initialized from the pre-trained language model, and \( A \) corresponds to the Advantage function, defined as follows \ref{eq:advantage}:

\begin{equation}
    A_t = \sum_{k=t}^{T} \gamma^{k-t} \left( r_k + \gamma V(s_{k+1}) - V(s_k) \right)
    \label{eq:advantage}
\end{equation}

Here r denotes the reward, &gamma;	 is the discount factor, and V stands for the value function. The value function in our experiment is a linear layer on top of the final embedding, which gives us a single number.

After computing the loss, the optimizer takes a single step to improve the reward. We employ an AdamW optimizer with a learning rate of 2e-5.

<h2> Reward Model </h2>

The reward function plays the most important role in training the model; a better reward function means a better model. Hence to calculate the reward for the generated summaries, we begin by tokenizing the generated summary into individual sentences. Next, we employ a pre-trained model, specifically trained on the Natural Language Inference task (RoBERTa), to determine the entailment relationship between each sentence and the original article. We assign a reward of one if a sentence and the article are "entailed," zero if they are "neutral," and negative one if they are in "contradiction." The reward for the summary is computed as the average of the rewards assigned to its constituent sentences. Finally, this overall reward is utilized to fine-tune the model.