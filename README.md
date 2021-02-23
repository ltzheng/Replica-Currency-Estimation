# Replica-Currency-Estimation

## Problem Setting

Sharing the history among different replicas is expensive, instead of that, for each replica we learn a model based on its update history and share the model with other replicas.

### Challenges

1) Local learning at each model is hard given the sparsity and incompleteness at each node if the distributions are not uniform across nodes, this will cause problems in the local learning. 

2) Combining the models (ensemble learning) will have to consider some loss from the ideal case when you have a model that has access to everything. 

## Simulated Scenarios

1. Single unavailable replica with uniform/exponential/poisson distribution respectively
2. Network failure: communication issue caused by unstable network, and in this case, the replica maybe unresponsive to some certain queries but still could answer other queries.
3. Node partition
4. Multiple unavailable replicas

## Currency Definition

Not just about the time, also about how many replicas has the current value. Each replica has a correct and incomplete updates history,
and we learn from the history in process model, and this model is shared with other replicas. For each replica, we can subscribe the individual models from other replicas and example them locally. And we can use the example to infer the complete history.

## Model

Consider an autoregressive model, which predicts the occurrence of next update according to the previous adjacent updates. There are global and local algorithms here.

To run the model with default setting:
```bash
python average.py
```

The get the description of hyperparameters:
```bash
python average.py --help
```