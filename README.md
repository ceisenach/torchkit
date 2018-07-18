# Natural Actor-Critic for Exponential Families
Implementation of natural actor-critic algorithms for exponential family distributions.


# Package Structure

## algorithms
### Natural AC
Natural actor-critic for exponential family.


### TRPO Implementation
Originally based on ikostrikov's port of John Schulman's code; mostly re-factored and re-implemented now.



## autodiff
Light wrapper around pytorch to implement auto-differentiation of vector valued functions.




## bin
Contains script to run training, e.g.
```
python bin/train.py --co -N 100
```