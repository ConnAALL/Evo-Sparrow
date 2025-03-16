# Evolutionary Optimization of Deep Learning Agents for Sparrow Mahjong

*Jim O'Connor - Derin Gezgin - Gary B. Parker*

This repository has the code for the paper *Evolutionary Optimization of Deep Learning Agents for Sparrow Mahjong*, which is currently under review for IEEE COG 2025. I will add the paper here if it gets published to give more detail on the project. 

There are three main folders you need to use in your training and evaluation pipeline.

### Training

This folder has the project's main training code. To start a test run, you can run the bash script **run_10_tests.sh**, which will automatically run 10 subsequent training runs. 

### Data Processing & Plotting

This folder has the code for viewing the progress of a training run. It has a script for merging SQLite databases, a script for processing the data for plotting, and another script for plotting the data using the SciencePlots family. 

### Benchmarking

After a model is trained, you can benchmark the weight file. The results will be saved into a .txt file.
