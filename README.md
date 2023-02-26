# Adaptive-saturated RNN: Remember more with less instability
This repository hosts the PyTorch code to implement the paper: [this will be updated upon acceptance]

For good reproducibility, we suggest to set up Pytorch 1.13.1 on a nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu18.04 Docker on [vast.ai](vast.ai).
If you find the paper or the source code useful to your projects, please support our works by citing the [bibtex](this will be updated upon acceptance): [this will be updated upon acceptance]

## Abstract
Orthogonal parameterization has offered a compelling solution to the vanishing gradient problem (VGP) in recurrent neural networks (RNNs). Thanks to orthogonal parameters and non-saturated activation functions, gradients in such models are constrained to unit norms. On the other hand, although the traditional vanilla RNN have been observed to possess higher memory capacity, they suffer from the VGP and perform badly in many applications. This work connects the two aforementioned approaches by proposing Adaptive-Saturated RNNs (asRNN), a variant that dynamically adjusts the saturation level between the two. Consequently, asRNN enjoys both the capacity of a vanilla RNN and the training stability of orthogonal RNNs. Our experiments show encouraging results of asRNN on challenging sequence learning benchmarks compared to several strong competitors.
## Model Architecture
*Formulation* We formally define the hidden cell of asRNN as:
$$h_t = W_f^{-1}\mathrm{tanh}(W_f(W_{xh}x_{t}+W_{hh}h_{t-1} + b)),$$
where $W_f = U_fD_f$, $U_f$ and $W_{hh}$ are parametrized orthogonal according to the [expRNN](https://arxiv.org/abs/1901.08428) paper, and ${[D_f]}_{i,j} = \delta_{ij}|{[P_f]}_{i,j}| + \epsilon$ for $\epsilon > 0$ is a hyperparameter.

Initially, $[P_f]_{i,j}\sim U(x;a,b)$ where $a, b$ are hyperparameters.
## Hyperparameters