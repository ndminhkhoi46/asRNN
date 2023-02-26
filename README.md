# Adaptive-saturated RNN: Remember more with less instability

This repository hosts the PyTorch code to implement the paper: [this will be updated upon acceptance]

For good reproducibility, we suggest setting up Pytorch 1.13.1 on a nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu18.04 Docker on [vast.ai](vast.ai). Our experiments used a RTX 3090 GPU.

If you find the paper or the source code useful to your projects, please support our works by citing the [bibtex](this will be updated upon acceptance): [this will be updated upon acceptance]

## Abstract

Orthogonal parameterization has offered a compelling solution to the vanishing gradient problem (VGP) in recurrent neural networks (RNNs). Thanks to orthogonal parameters and non-saturated activation functions, gradients in such models are constrained to unit norms. On the other hand, although the traditional vanilla RNN have been observed to possess higher memory capacity, they suffer from the VGP and perform badly in many applications. This work connects the two aforementioned approaches by proposing Adaptive-Saturated RNNs (asRNN), a variant that dynamically adjusts the saturation level between the two. Consequently, asRNN enjoys both the capacity of a vanilla RNN and the training stability of orthogonal RNNs. Our experiments show encouraging results of asRNN on challenging sequence learning benchmarks compared to several strong competitors.

## Model Architecture

*Formulation* We formally define the hidden cell of asRNN as:
$$h_t = W_f^{-1}\mathrm{tanh}(W_f(W_{xh}x_{t}+W_{hh}h_{t-1} + b)),$$
where $W_f = U_fD_f$, $U$ and $W_{hh}$ are parametrized orthogonal according to the [expRNN](https://arxiv.org/abs/1901.08428) paper, and $D_f$ is strictly positive diagonal.

More details of the implementation can be found in Appendix A.4 of our paper.

## Hyperparameters

For hyperparameters $a, b, \epsilon$, refer Table~3 in the paper.
To setup expRNN or scoRNN experiments, refer to this [repository](https://github.com/Lezcano/expRNN).

### Common setting

| Data | Classification | Batch first | No. Epoch/Iteration | Seed | Source |
| --- | --- | --- | --- | --- | --- |
| MNIST | Many-to-one | True | $70$ | $5544$ | http://yann.lecun.com/exdb/mnist/ |
| Copying Memory | Many-to-many | True | $4000$ | $5544$ | https://github.com/ajithcodesit/lstm_copy_task |
| Penn Treebank (char Level) | Many-to-many | False | $100$ | 5 repeated trials | http://www.fit.vutbr.cz/~imikolov/rnnlm/ |

- Optimizer is RMSprop.
- Forget gates of LSTM are initiated with $1$.

- Batch size is $128$  for every experiment. PTB-c has evaluation batch size of $10$.
- Benchmarks at PTB-c task *do not* use gradient clipping.

### sequential MNIST task

- scoRNN: $\rho = \frac{1}{10}$

| Model | $d_h$ | Learning Rate | Recurrent Learning Rate | $\ln(W_{hh})$ init | $\alpha$ | Gradient Clipping |
| --- | --- | --- | --- | --- | --- | --- |
| asRNN | $122$ | $7\times10^{-4}$ | $7\times10^{-5}$ | Cayley | $0.99$ | $10$ |
| expRNN | $170$ | $7\times10^{-4}$ | $7\times10^{-5}$ | Cayley  | $0.99$ | $-$ |
| scoRNN | $170$ | $10^{-3}$ | $10^{-4}$ | Cayley | $0.99$ | $-$ |
| asRNN | $257$ | $5\times10^{-4}$ | $5\times10^{-5}$ | Cayley | $0.99$ | $10$ |
| expRNN | $360$ | $5\times10^{-4}$ | $5\times10^{-5}$ | Cayley | $0.99$ | $-$ |
| scoRNN | $360$ | $10^{-3}$ | $10^{-5}$ | Cayley | $0.99$ | $-$ |
| LSTM | $128$ | $10^{-3}$ | $-$ | $-$ | $0.9$ | $1$ |
| asRNN | $364$ | $3\times10^{-4}$ | $3\times10^{-5}$ | Cayley | $0.99$ | $10$ |
| expRNN | $512$ | $3\times10^{-4}$ | $3\times10^{-5}$ | Cayley | $0.99$ | $-$ |
| scoRNN | $512$ | $10^{-3}$ | $10^{-5}$ | Cayley | $0.99$ | $-$ |

### permuted MNIST task

- scoRNN: $\rho = \frac{1}{2}$

| Model | $d_h$ | Learning Rate | Recurrent Learning Rate | $\ln(W_{hh})$ init | $\alpha$ | Gradient Clipping |
| --- | --- | --- | --- | --- | --- | --- |
| asRNN | $122$ | $10^{-3}$ | $10^{-4}$ | Cayley | $0.99$ | $10$ |
| expRNN | $170$ | $10^{-3}$ | $10^{-4}$ | Cayley  | $0.99$ | $-$ |
| scoRNN | $170$ | $10^{-3}$ | $10^{-4}$ | Cayley  | $0.99$ | $-$ |
| asRNN | $257$ | $7\times10^{-4}$ | $7\times10^{-5}$ | Cayley  | $0.99$ | $10$ |
| expRNN | $512$ | $5\times10^{-4}$ | $5\times10^{-5}$ | Cayley  | $0.99$ | $-$ |
| scoRNN | $360$ | $10^{-3}$ | $10^{-5}$ | Cayley | $0.99$ | $-$ |
| LSTM | $128$ | $10^{-3}$ | $-$ | $-$ | $0.9$ | $1$ |
| asRNN | $364$ | $5\times10^{-4}$ | $5\times10^{-5}$ | Cayley  | $0.99$ | $10$ |
| expRNN | $360$ | $7\times10^{-4}$ | $7\times10^{-5}$ | Cayley  | $0.99$ | $-$ |
| scoRNN | $512$ | $10^{-3}$ | $10^{-5}$ | Cayley  | $0.99$ | $-$ |

### Copying Memory tasks

- scoRNN: $\rho = \frac{1}{10}$

| Model | $d_h$ | Learning Rate | Recurrent Learning Rate | $\ln(W_{hh})$ init | $\alpha$ | Gradient Clipping |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | $68$ | $0.001$ | $-$ | $-$ | $0.9$ | $1$ |
| scoRNN | $190$ | $10^{-3}$ | $10^{-4}$ | Henaff  | $0.99$ | $-$ |
| expRNN | $190$ | $2\times10^{-4}$ | $2\times10^{-5}$ | Henaff  | $0.99$ | $-$ |
| asRNN | $138$ | $7\times10^{-4}$ | $7\times10^{-5}$ | Henaff  | $0.99$ | $10$ |

### Penn Treebank Character-level Prediction ($T_{PTB}=150$)

| Model | $d_h$ | Learning Rate | Recurrent Learning Rate | $W_{hh}$/$\ln(W_{hh})$ init | $\alpha$ |
| --- | --- | --- | --- | --- | --- |
| LSTM | $475$ | $0.001$ | $-$ | $-$ | $0.99$ |
| asRNN | $1024$ | $0.005$ | $0.0005$ | Cayley | $0.9$ |
| expRNN | $1386$ | $0.005$ | $0.0001$ | Cayley | $0.9$ |
| RNN initialized orthogonal | $1024$ | $0.0001$ | $-$ | Random orthogonal | $0.9$ |
| RNN | $1024$ | $10^{-5}$ | $-$ | Glorot Normal | $0.9$ |

### Penn Treebank Character-level Prediction ($T_{PTB}=300$)

| Model | $d_h$ | Learning Rate | Recurrent Learning Rate | $W_{hh}$ init | $\alpha$ |
| --- | --- | --- | --- | --- | --- |
| LSTM | $475$ | $0.003$ | $-$ | $-$ | $0.9$ |
| asRNN | $1024$ | $0.005$ | $0.0001$ | Cayley | $0.9$ |
| expRNN | $1386$ | $0.005$ | $0.0001$ | Cayley | $0.9$ |
| RNN initialized orthogonal | $1024$ | $0.0001$ | $-$ | Cayley | $0.9$ |
| RNN | $1024$ | $10^{-5}$ | $-$ | Glorot Normal | $0.9$ |

## Experiment Results

### Best test accuracy on pixelated MNIST tasks

| Model | #PARAMS | $d_h$ | sMNIST | pMNIST |
| --- | --- | --- | --- | --- |
| asRNN | $16\times10^3$ | $122$ | $\bf98.89\%$ | $\bf95.41\%$ |
| expRNN | $16\times10^3$ | $170$ | $98.0\%$ | $94.9\%$ |
| scoRNN | $16\times10^3$ | $170$ | $97.2\%$ | $94.8\%$ |
| asRNN | $69\times10^3$ | $257$ | $\bf99.21\%$ | $\bf96.88\%$ |
| expRNN | $69\times10^3$ | $360$ | $98.4\%$ | $96.2\%$ |
| scoRNN | $69\times10^3$ | $360$ | $98.1\%$ | $95.9\%$ |
| LSTM | $69\times10^3$ | $128$ | $81.9\%$ | $79.5\%$ |
| asRNN | $137\times10^3$ | $364$ | $\bf99.3\%$ | $\bf96.96\%$ |
| expRNN | $137\times10^3$ | $512$ | $98.7\%$ | $96.6\%$ |
| scoRNN | $137\times10^3$ | $512$ | $98.2\%$ | $96.5\%$ |

### Training Cross Entropy on Copying Memory tasks

![Recall Length $K =10$, Delay Length $L = 1000$.](Hyperparameters%20d2b70a8fb40c4ef59ffd5d05ed5d3d1c/copy_1000.png)

Recall Length $K =10$, Delay Length $L = 1000$.

![Recall Length $K =10$, Delay Length $L = 2000$.](Hyperparameters%20d2b70a8fb40c4ef59ffd5d05ed5d3d1c/copy_2000.png)

Recall Length $K =10$, Delay Length $L = 2000$.

### Bit-per-character results on test set of Penn Treebank Character-level Prediction tasks

| Model | #PARAMS | $d_h$ | $T_{BPTT}=150$ | $T_{BPTT}=300$ |
| --- | --- | --- | --- | --- |
| LSTM | $1.32\times10^6$ | $475$ | $\bf 1.41 \pm 0.005$ | $\bf 1.43\pm0.004$ |
| asRNN | $1.32\times10^6$ | $1024$ | $1.46 ± 0.006$ | $1.49 \pm 0.005$ |
| expRNN | $1.32\times10^6$ | $1386$ | $1.49\pm 0.008$ | $1.52 \pm 0.001$ |
| orth-RNN | $1.32\times10^6$ | $1024$ | $1.62\pm 0.004$ | $1.66 \pm 0.006$ |
| RNN | $1.32\times10^6$ | $1024$ | $2.89\pm 0.002$ | $2.90 \pm 0.0016$ |

## To Do:

- Set up a Hyperparameter Tuner.