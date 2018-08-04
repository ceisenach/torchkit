# Natural Actor-Critic for Exponential Families
Implementation of natural actor-critic algorithms for exponential family distributions.


# Package Structure

## algorithms
### A2C
A synchronous version of A3C. Provided as a comparison for NACGauss and TRPO implementations

### Natural AC
Natural actor-critic for exponential family. NACGauss is the Natural Actor Critic for the Gaussian family. So far, it supports Gaussian distributions with diagonal covariance structure. 



### TRPO Implementation
TRPO is a PyTorch implementation of John Schulman's original implementation.

TRPO_v2 is a PyTorch implementation of OpenAI Baselines TRPO_MPI. Performance of our implementation is the same as baselines.


## autodiff
Light wrapper around pytorch to implement auto-differentiation of vector valued functions.



## bin
Contains script to run training, e.g.
```
python bin/train.py --co -N 100
```
More examples
```
 python bin/train.py -V 1 -N 10 --ack "{'hidden_layers' :  2}" --lr 0.01  -u 1e4  -co
```
Flag '--co' indicates to log output to console instead of to file. Can be useful for debugging. Passing in seed of -1 means that no seed is used.


Make seaborn confidence bands plot
```
python bin/plot.py -d res/exp_angular/ res/exp_gauss res/exp_gauss1D/ -n Angular MVG 1DG -o res --plot sns
```
To use plotting, all runs of the same experiment should be in the same directory