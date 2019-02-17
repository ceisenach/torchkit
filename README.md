# Torchkit

This repository contains a collection (or kit) of tools for Deep RL built with PyTorch. They include

 - an autodifferentiation wrapper around PyTorch for vector valued functions
 - efficient implementations of TRPO and natural policy gradient for high dimensional action spaces
 - running batches of RL experiments
 - creating pretty plots using seaborn
 - some additional RL environments

## Related Publications

[[1] Carson Eisenach, Haichuan Yang, Ji Liu, and Han Liu. "Marginal Policy Gradients: A Unified Family of Estimators for Bounded Action Spaces with Applications". *International Conference on Learning Representations (ICLR'19)*, New Orleans, USA, 2019.](http://princeton.edu/~eisenach/publication/marginal-policy-gradients/)

[[2] Carson Eisenach and Zhuoran Yang. "Exponential Family Policy Gradient". Technical Report, Princeton University, Princeton, NJ, 2018.](http://princeton.edu/~eisenach/publication/exponential-family-npg/)

## Credits
### Contributors
The following individuals have contributed either to the code or to the underlying ideas
 - Carson Eisenach
 - Haichuan Yang (contributions to the angular policy gradient implementation)
 - Zhuoran Yang (contributions to the exponential family natural policy gradient)


### Other
The TRPO implementations implementations are based on (or ported from)
 - [Open AI Baselines](https://github.com/openai/baselines)
 - [John Schulman's Original TRPO](https://github.com/joschu/modular_rl)
 - [Ilya Kostrikiov's PyTorch Implementation](https://github.com/ikostrikov/pytorch-trpo/)

The Platform2D environment and AngularGaussian policy are adapted from https://github.com/ceisenach/MPG

## Notes
Compatibility with Pytorch 1.0 confirmed.

# Package Structure
The Torchkit package is comprised of the following sub-packages:
 - [algorithm](#algorithm)
 - [autodiff](#autodiff)
 - [bin](#bin)
 - [environment](#environment)
 - [experiments](#experiments)
 - [model](#model)
 - [optimize](#optimize)
 - [policy](#policy)
 - [running_stat](#running_stat)
 - [sampler](#sampler)


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
This module contains scripts to perform basic functionality essential to deep RL research. This includes:
 - running batches of experiments (`run_experiment.py`)
 - training a single model (`train.py`)
 - scripts to create pretty plots using seaborn (`plot.py`).

In general, you can pass the argument `-h` to view the help. The default training parameters are tailored to TRPO, override them from the command line if you are using a different algorithm.

### Examples
Running training:
```
python bin/train.py --co -N 100
 python bin/train.py -V 1 -N 10 --nk "{'hidden_layers' :  2}" --lr 0.01  -u 1e4  -co
```
Flag '--co' indicates to log output to console instead of to file. Can be useful for debugging. Passing in seed of -1 means that no seed is used.

For A2C
```
python bin/train.py --e Platform2D-v1 --co -a A2C -p AngularGaussML --nt GaussianML Value --lra 0.001 -N 25 -u 1e5
```

For TRPO
```
 python bin/train.py --e Platform2D-v1 --co -N 1000 -a TRPO
```
For NAC
```
 python bin/train.py --e Platform2D-v1 --co -N 32 -a NAC
 python bin/train.py --e Platform2D-v1 --co -N 128 -a NAC --lrc 0.01 --lra 0.1
 python bin/train.py --e InvertedPendulum-v2 -V 1 -N 512 -a NAC --lrc 0.001 --lra 0.001 -u 10000 --co
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
Some new tasks to test out RL algorithms:
 - Platform2D (from [1])
 - Optimial production problems (from [2])


## experiments
While not a python package, we describe the experiment runner mechanism in detail.  The experiment runner script, `bin/run_experiment.py`, is used to run batch experiments using a fixed set of seeds (for reproducibility).

In this directory, there are several `.json` files containing *experiment descriptions* along with a set of random seeds (`seeds.txt`). The results of these experiments can be found in [2].


## model
Models for value and policy nets. Intended for use with the autodiff package, but can be used without it as well (fully compatible with standard pytorch).



## optimize
Optimization routines that include L-BFGS and the conjugate gradient method (used to compute the Fisher-vector-product)


## policy
An implementation of several common distributions including: multivariate Gaussian, Beta and Gamma. For the multivariate Gaussian, several different parametrizations are provided.

Policy objects can be used to sample from the distribution as well as compute quantities like the negative log-likelihood and KL divergence.

To add new types of policies, simply implement the interface provided by `BasePolicy`.

## running_stat
Can be used to normalize input to either the policy or value nets.

## sampler
Generate samples from an environment given the current policy. It can also sample from multiple independent copies of the same env

## utils
Various utility functions for running experiments.

# Roadmap
This torchkit is a work-in-progress. Some improvements on the roadmap include:
 - clean up the autodiff API
 - more RL environments
