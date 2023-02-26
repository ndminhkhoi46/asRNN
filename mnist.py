import signal
exit_signal = False
def handler(signum, frame):
    global exit_signal
    exit_signal = True   
signal.signal(signal.SIGINT, handler) 

import subprocess
subprocess.call(['pip', 'install', 'scipy'])
subprocess.call(['pip', 'install', 'pandas'])
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import numpy as np
import random
import time
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from datetime import datetime
import sys
import argparse
import pandas as pd
from torch.nn import Embedding
root_path = './'
expr_path = root_path + 'Benchmark/MNIST/'
import os
os.makedirs(expr_path, exist_ok=True)
import_path = [root_path + subpath for subpath in ['sources/', 'sources/expRNN/']]
for path in import_path:
  sys.path.append(path)

from model_loader import Model
from custom_modules import asRNN
from orthogonal import modrelu
from trivializations import cayley_map, expm
from initialization import henaff_init_, cayley_init_
from data_module import MNISTDataModule
sys.argv = ['']
parser = argparse.ArgumentParser(description='MNIST Task')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=122)
parser.add_argument('--epochs', type=int, default=70)
parser.add_argument('--rmsprop_lr', type=float, default=7e-4)
parser.add_argument('--rmsprop_constr_lr', type=float, default=7e-5)
parser.add_argument('--alpha', type=float, default=0.99)
parser.add_argument('--clip_norm', type=float, default=10)
parser.add_argument("--permute", action="store_true", default = False)
parser.add_argument("-m", "--mode",
                    choices=["exprnn", "dtriv", "cayley", "lstm", "rnn"],
                    default="exprnn",
                    type=str)
parser.add_argument("--nonlinear",
                    choices=["asrnn", "modrelu"],
                    default="asrnn",
                    type=str)
parser.add_argument("-a", type=float, default=2e-2)
parser.add_argument("-b", type=float, default=2e-2)
parser.add_argument("--eps", type=float, default=1e-2)
parser.add_argument('--K', type=str, default="100", help='The K parameter in the dtriv algorithm. It should be a positive integer or "infty".')
parser.add_argument("--init",
                    choices=["cayley", "henaff"],
                    default="cayley",
                    type=str)
parser.add_argument('--random-seed', type=int, default=5544,
                    help='random seed')
parser.add_argument('--rho_rat_den', type=int, default=10)
parser.add_argument('--forget_bias', type=int, default=1)
parser.add_argument('--emsize', type=int, default=0,
                    help='size of word embeddings')
#Setting
#https://arxiv.org/pdf/1901.08428.pdf
#https://arxiv.org/abs/1905.12080
args = parser.parse_args()
print(args)

# Fix seed across experiments
# Same seed as that used in "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform"
# https://github.com/SpartinStuff/scoRNN/blob/master/scoRNN_copying.py#L79
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

#Deterministic training
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8" #increase library footprint in GPU memory by approximately 24MiB


sequence_length = 784
hidden_size = args.hidden_size
batch_size  = args.batch_size
epochs      = args.epochs
batch_first = True
many_to_many = False

if args.init == "cayley":
    init =  cayley_init_
elif args.init == "henaff":
    init = henaff_init_

if args.K != "infty":
    args.K = int(args.K)
if args.mode == "exprnn":
    mode = "static"
    param = expm
elif args.mode == "dtriv":
    # We use 100 as the default to project back to the manifold.
    # This parameter does not really affect the convergence of the algorithms, even for K=1
    mode = ("dynamic", args.K, 100)
    param = expm
elif args.mode == "cayley":
    mode = "static"
    param = cayley_map
else:
    mode = None
    param = None
if args.nonlinear == "asrnn":
    nonlinearity = asRNN(hidden_size, torch.zeros_like, mode, param, args.a, args.b, args.eps)
elif args.nonlinear == "modrelu":
    nonlinearity = modrelu(hidden_size)
#Setting up data module
# Load data   
dataset_path = root_path + 'Dataset/MNIST/'
if args.permute:
  datamodule = MNISTDataModule(sequence_length, 784 // sequence_length, batch_size, dataset_path, permute_seed = 92916)
else:
  datamodule = MNISTDataModule(sequence_length, 784 // sequence_length, batch_size, dataset_path, permute_seed = None)

if args.emsize > 0:
  embed_layer = Embedding(datamodule.input_size, args.emsize) # Token2Embeddings
  input_size = args.emsize
else:
  embed_layer = None
  input_size = datamodule.input_size

#Initialize Model
model = Model(input_size, hidden_size, datamodule.output_size, nonlinearity, initializer_skew = init,
              mode = mode, param = param, args=args, embed_layer=embed_layer, batch_first=batch_first).to(device)
              
##Initialize Optimizers
unconstrained_parameters = []
constrained_parameters = []
for name, p in model.named_parameters():
  if any(map(name.__contains__, ['recurrent_kernel'])):
      constrained_parameters.append(p)
  else:
      unconstrained_parameters.append(p)
rmsprop_optim = torch.optim.RMSprop([
                  {'params': unconstrained_parameters},
                  {'params': constrained_parameters, 'lr': args.rmsprop_constr_lr}
              ], lr=args.rmsprop_lr, alpha = args.alpha)
model.optim_list = [rmsprop_optim]

column_names = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'train_time']
train_log = pd.DataFrame(columns = column_names)

def pred(model, datamodule, train = True):
  global device, exit_signal
  total_loss = 0.0
  total_correct = 0.0
  sample_num = 0

  data_enum = datamodule.enum(train)
  data_len = datamodule.len(train)
  for batch_x, batch_y in data_enum:
      batch_x, batch_y = batch_x.to(device).view(-1, datamodule.sequence_length, datamodule.input_size), batch_y.to(device)
      state = model.init_state(batch_x.size(1-model.dim_T))
      logits, _ = model(batch_x, state)
      batch_pred = torch.select(logits, model.dim_T, -1)
      loss = model.loss(batch_pred, batch_y)
      if train:
        model.zero_grad(set_to_none = True)
        loss.backward()
        if model.args.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.args.clip_norm)
        for optim in model.optim_list:
            if optim:
                optim.step()
      with torch.no_grad():
        total_loss += loss.item()
        batch_correct = model.correct(batch_pred, batch_y).item()
      total_correct += batch_correct
      sample_num += batch_x.size(0)
      if train: #For tracking training progress
        print('train: epoch {} ({:.2f}%)| loss: {:.3f}| acc: {:.2f}'.format(epoch, 100*sample_num/datamodule.train_len, loss.item(),
            batch_correct/batch_x.size(1-model.dim_T)))
      if exit_signal: #Safe interruption to avoid GPU memory buffer overflow
        break
        
  total_loss /=  len(data_enum)
  total_acc = 100 * total_correct / data_len
  return total_loss, total_acc

#Training
model.train()
best_test_acc = 0
for epoch in range(epochs):
  model.train()
  start = time.time()
  train_loss, train_acc = pred(model, datamodule, train = True)
  train_time = time.time()-start
  model.eval()
  with torch.no_grad():
      test_loss, test_acc = pred(model, datamodule, train = False)
  best_test_acc = max(best_test_acc, test_acc)                                                                                 

  new_row = pd.Series({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, 'test_loss': test_loss, 'test_acc': test_acc, 'train_time': train_time})
  train_log = model.update_log_df(new_row, train_log, expr_path+'/train_log.pkl')
  
  if exit_signal:
    print('-' * 89)
    print('Exiting from training early')
    break
    
  print('=' * 89)
  print('epoch {}: train_loss {:.3f}| test_loss {:.3f}| test_acc {:.3f}%| best_test_acc {}%| train_time {:.3f}s'.format(epoch, train_loss,
  test_loss, test_acc, best_test_acc, train_time)) 
  print('=' * 89)   
  
#Ring a bell to notice the completion
print('\007')