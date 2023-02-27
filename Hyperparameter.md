# Hyperparameter

For hyperparameters $a, b, \epsilon$, refer Table $3$ in the paper.
To setup expRNN or scoRNN experiments, please refer [https://github.com/Lezcano/expRNN](https://github.com/Lezcano/expRNN) repository.

### Common setting

| Data | Classification | Batch first | No. Epoch/Iteration | Seed | Source |
| --- | --- | --- | --- | --- | --- |
| MNIST | Many-to-one | True | $70$ | $5544$ | http://yann.lecun.com/exdb/mnist/ |
| Copying Memory | Many-to-many | True | $4000$ | $5544$ | https://github.com/ajithcodesit/lstm_copy_task |
| Penn Treebank (char Level) | Many-to-many | False | $100$ | 5 repeated trials | http://www.fit.vutbr.cz/~imikolov/rnnlm/ |

- Optimizer is RMSprop
- Forget gates of LSTM are initiated with $1$

- Batch size is $128$  for every experiment PTB-c has evaluation batch size of $10$
- Benchmarks at PTB-c task *do not* use gradient clipping

### sequential MNIST task

- scoRNN: $\rho = \frac{1}{10}$

| Model | hidden_size | Learning Rate | Recurrent Learning Rate | $\ln(W_{hh})$ init | $\alpha$ | Gradient Clipping |
| --- | --- | --- | --- | --- | --- | --- |
| asRNN | $122$ | $7\times10^{-4}$ | $7\times10^{-5}$ | Cayley | $0.99$ | $10$ |
| expRNN | $170$ | $7\times10^{-4}$ | $7\times10^{-5}$ | Cayley  | $0.9$ | $-$ |
| scoRNN | $170$ | $7\times10^{-4}$ | $7\times10^{-5}$ | Cayley | $0.9$ | $-$ |
| asRNN | $257$ | $5\times10^{-4}$ | $5\times10^{-5}$ | Cayley | $0.99$ | $10$ |
| expRNN | $360$ | $5\times10^{-4}$ | $5\times10^{-5}$ | Cayley | $0.9$ | $-$ |
| scoRNN | $360$ | $5\times10^{-4}$ | $5\times10^{-5}$ | Cayley | $0.99$ | $-$ |
| LSTM | $128$ | $10^{-3}$ | $-$ | $-$ | $0.9$ | $1$ |
| asRNN | $364$ | $3\times10^{-4}$ | $3\times10^{-5}$ | Cayley | $0.99$ | $10$ |
| expRNN | $512$ | $3\times10^{-4}$ | $3\times10^{-5}$ | Cayley | $0.9$ | $-$ |
| scoRNN | $512$ | $3\times10^{-4}$ | $3\times10^{-5}$ | Cayley | $0.9$ | $-$ |

### permuted MNIST task

- scoRNN: $\rho = \frac{1}{2}$

| Model | hidden_size | Learning Rate | Recurrent Learning Rate | $\ln(W_{hh})$ init | $\alpha$ | Gradient Clipping |
| --- | --- | --- | --- | --- | --- | --- |
| asRNN | $122$ | $10^{-3}$ | $10^{-4}$ | Cayley | $0.99$ | $10$ |
| expRNN | $170$ | $10^{-3}$ | $10^{-4}$ | Cayley  | $0.9$ | $-$ |
| scoRNN | $170$ | $10^{-3}$ | $10^{-4}$ | Cayley  | $0.9$ | $-$ |
| asRNN | $257$ | $7\times10^{-4}$ | $7\times10^{-5}$ | Cayley  | $0.99$ | $10$ |
| expRNN | $512$ | $5\times10^{-4}$ | $5\times10^{-5}$ | Cayley  | $0.9$ | $-$ |
| scoRNN | $360$ | $5\times10^{-4}$ | $5\times10^{-5}$ | Cayley | $0.9$ | $-$ |
| LSTM | $128$ | $10^{-3}$ | $-$ | $-$ | $0.9$ | $1$ |
| asRNN | $364$ | $5\times10^{-4}$ | $5\times10^{-5}$ | Cayley  | $0.99$ | $10$ |
| expRNN | $360$ | $7\times10^{-4}$ | $7\times10^{-5}$ | Cayley  | $0.9$ | $-$ |
| scoRNN | $512$ | $7\times10^{-4}$ | $7\times10^{-5}$ | Cayley  | $0.9$ | $-$ |

### Copying Memory tasks

- scoRNN: $\rho = \frac{1}{2}$

| Model | hidden_size | Learning Rate | Recurrent Learning Rate | $\ln(W_{hh})$ init | $\alpha$ | Gradient Clipping |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | $68$ | $10^{-3}$ | $-$ | $-$ | $0.9$ | $1$ |
| scoRNN | $190$ | $2\times10^{-4}$ | $2\times10^{-5}$ | Henaff  | $0.9$ | $-$ |
| expRNN | $190$ | $2\times10^{-4}$ | $2\times10^{-5}$ | Henaff  | $0.9$ | $-$ |
| asRNN | $138$ | $2\times10^{-4}$ | $2\times10^{-5}$ | Henaff  | $0.9$ | $10$ |

### Penn Treebank Character-level Prediction ($T_{PTB}=150$)

| Model | hidden_size | Learning Rate | Recurrent Learning Rate | $\ln(W_{hh})$ init | $\alpha$ |
| --- | --- | --- | --- | --- | --- |
| LSTM | $475$ | $10^{-3}$ | $-$ | $-$ | $0.99$ |
| asRNN | $1024$ | $10^{-3}$ | $10^{-4}$ | Cayley | $0.9$ |
| expRNN | $1386$ | $5\times10^{-3}$ | $10^{-4}$ | Cayley | $0.9$ |

### Penn Treebank Character-level Prediction ($T_{PTB}=300$)

| Model | hidden_size | Learning Rate | Recurrent Learning Rate | $\ln{W_{hh}}$ init | $\alpha$ |
| --- | --- | --- | --- | --- | --- |
| LSTM | $475$ | $3\times10^{-3}$ | $-$ | $-$ | $0.9$ |
| asRNN | $1024$ | $10^{-3}$ | $10^{-3}$ | Cayley | $0.9$ |
| expRNN | $1386$ | $5\times10^{-3}$ | $10^{-4}$ | Cayley | $0.9$ |