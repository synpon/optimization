# Learning Gradient Descent with Synthetic Objectives

Research project to learn generalizable optimizers for neural networks from synthetic objective functions.    


**Prerequisites**
 - TensorFlow    
 - Numpy    


Run main.py to train the optimizer and compare.py to test it against SGD and Adam.    

So far the best results have been obtained by running supervised learning (as opposed to reinforcement learning) with an additional penalty for oscillation. Using an asymmetric loss function that heavily penalises increases in the loss function has a similar effect. 

