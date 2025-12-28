# Evolutionary Optimization of Deep Learning Agents for Sparrow Mahjong

[Jim O'Connor](https://oconnor.digital.conncoll.edu) | [Derin Gezgin](https://deringezgin.github.io) | [Gary B. Parker](https://oak.conncoll.edu/parker/)

*Published in AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, 2025*

This repository has the code for the paper *Evolutionary Optimization of Deep Learning Agents for Sparrow Mahjong*.

There are three main folders you need to use in your training and evaluation pipeline.

## Training

This folder has the project's main training code. To start a test run, you can run the bash script **run_10_tests.sh**, which will automatically run 10 subsequent training runs. 

## Data Processing & Plotting

This folder has the code for viewing the progress of a training run. It has a script for merging SQLite databases, a script for processing the data for plotting, and another script for plotting the data using the SciencePlots family. 

## Benchmarking

After a model is trained, you can benchmark the weight file. The results will be saved into a .txt file.

> The project uses the pgx library to simulate the game. We are using a slightly modified version of the library for saving the scores. Please use the library folder that already exists in the project. The scripts are adjusted to look at the upper-level folder when importing the library, so it should work as it is. 

## Citation

```bibtex
@inproceedings{10.1609/aiide.v21i1.36833,
author = {O'Connor, Jim and Gezgin, Derin and Parker, Gary B.},
title = {Evolutionary optimization of deep learning agents for Sparrow Mahjong},
year = {2025},
isbn = {1-57735-904-6},
publisher = {AAAI Press},
url = {https://doi.org/10.1609/aiide.v21i1.36833},
doi = {10.1609/aiide.v21i1.36833},
abstract = {We present Evo-Sparrow, a deep learning-based agent for AI decision-making in Sparrow Mahjong, trained by optimizing Long Short-Term Memory (LSTM) networks using Covariance Matrix Adaptation Evolution Strategy (CMA-ES). Our model evaluates board states and optimizes decision policies in a non-deterministic, partially observable game environment. Empirical analysis conducted over a significant number of simulations demonstrates that our model outperforms both random and rule-based agents, and achieves performance comparable to a Proximal Policy Optimization (PPO) baseline, indicating strong strategic play and robust policy quality. By combining deep learning with evolutionary optimization, our approach provides a computationally effective alternative to traditional reinforcement learning and gradient-based optimization methods. This research contributes to the broader field of AI game playing, demonstrating the viability of hybrid learning strategies for complex stochastic games. These findings also offer potential applications in adaptive decision-making and strategic AI development beyond Sparrow Mahjong.},
booktitle = {Proceedings of the Twenty-First AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
articleno = {30},
numpages = {8},
location = {Edmonton, Alberta, Canada},
series = {AIIDE '25}
}
```
