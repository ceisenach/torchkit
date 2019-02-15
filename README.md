# Torchkit

This repository contains a collection (or kit) of tools for Deep RL built with PyTorch. They include
 - running batches of RL experiments
 - creating pretty plots using seaborn
 - an autodifferentiation wrapper around PyTorch for vector valued functions
 - some additional RL environments

## Contributors
The following individuals have contributed either to the code or to the underlying ideas
 - Carson Eisenach
 - Haichuan Yang (contributions to the angular policy gradient implementation)
 - Zhuoran Yang (contributions to the exponential family natural policy gradient)

## Related Publications

[1] Carson Eisenach, Haichuan Yang, Ji Liu, and Han Liu. "Marginal Policy Gradients: A Unified Family of Estimators for Bounded Action Spaces with Applications". *International Conference on Learning Representations (ICLR'19)*, New Orleans, USA, 2019.
[2] Carson Eisenach and Zhuoran Yang. "Exponential Family Policy Gradient". Technical Report, Princeton University, Princeton, NJ, 2018.

# Package Structure

## algorithm
### A2C
A synchronous version of A3C. Provided as a comparison for NACGauss and TRPO implementations


### Natural AC
Natural actor-critic for exponential family. NACGauss is the Natural Actor Critic for the Gaussian family. So far, it supports Gaussian distributions with diagonal covariance structure.


### TRPO Implementation
Two implementations are provided. These are pytorch ports of implementations by researchers at OpenAI. Specifically,
 - TRPO is a PyTorch implementation of John Schulman's original implementation.
 - TRPO_v2 is a PyTorch implementation of OpenAI Baselines TRPO_MPI. Performance of our implementation is the same as baselines.

For these implementations using running normalization (option --rstat) is very useful.


## autodiff
Light wrapper around pytorch to implement auto-differentiation of vector valued functions. This allows for *significantly* improved efficiency in high dimensional action spaces because the  Hessian can be factored as the product of several lower rank matrices.

### Implementation Notes
As a backend to the autodifferentiation engine, we provide both a Pytorch and pure Numpy implementations. The reason a pure numpy implementation is required is that PyTorch v0.40 has a memory leak when running on the CPU that we were unable to track down (some buffers were not getting cleaned up). The authors have not confirmed whether or not this issue still exists in the most recent PyTorch release.


## bin
THis module contains scripts to perform basic functionality essential to deep RL research. This includes:
 - scripts to run batches of experiments
 - scripts to create pretty plots using seaborn.

### Examples
Running training:
```
python bin/train.py --co -N 100
 python bin/train.py -V 1 -N 10 --nk "{'hidden_layers' :  2}" --lr 0.01  -u 1e4  -co
```
Flag '--co' indicates to log output to console instead of to file. Can be useful for debugging. Passing in seed of -1 means that no seed is used.

For TRPO
```
 python bin/train.py --e Platform2D-v1 --co -N 1000 -a TRPO
```
For NAC
```
 python bin/train.py --e Platform2D-v1 --co -N 32 -a NACGauss
 python bin/train.py --e Platform2D-v1 --co -N 128 -a NACGauss --lrc 0.01 --lra 0.1
 python bin/train.py --e InvertedPendulum-v2 -V 1 -N 512 -a NACGauss --lrc 0.001 --lra 0.001 -u 10000 --co
```
To use Beta policy instead of Gaussian, add the flags
```
-p Beta --nt PolicyBeta Value
```
Make seaborn confidence bands plot
```
python bin/plot.py -d res/exp_angular/ res/exp_gauss res/exp_gauss1D/ -n Angular MVG 1DG -o res --plot sns
```
To use plotting, all runs of the same experiment should be in the same directory

## environment


## policy


## utils

# Roadmap
This torchkit is a work-in-progress. Some improvements on the roadmap include:
 - clean up the autodiff API
 - more RL environments
