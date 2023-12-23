# Bayesian Synapse RNN
This repository contains an implementation of the RNN in the paper:

[**Synaptic Plasticity as Bayesian Inference**](https://www.nature.com/articles/s41593-021-00809-5)

Laurence Aitchison, Jannes Jegminat, Jorge Aurelio Menendez, Jean-Pascal Pfister, Alexandre Pouget & Peter E. Latham 

## Install 
To build the appropriate conda environment, run the following command:
```
conda env create -f environment.yml
```
To activate the environment, run:
```
conda activate bayes-rnn
```

## Run
To train the classical model, run
```
python ml.py
```
The difference between the target function $V_{tar}(t)$ and the network estimate $V(t)$ is plotted in 'outs.png'.