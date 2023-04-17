import signal
exit_signal = False
def handler(signum, frame):
    global exit_signal
    exit_signal = True   
signal.signal(signal.SIGINT, handler) 
import subprocess
subprocess.call(['pip', 'install', 'scipy'])
subprocess.call(['pip', 'install', 'pandas'])
import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import numpy as np
import random
import os
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
import sys
import argparse
import sys
import pandas as pd

root_path = './'
dataset_path = root_path + 'Dataset/PTB/'
expr_path = root_path + 'Benchmark/PTB/char-level/'
os.makedirs(expr_path, exist_ok=True)

import_path = [root_path + subpath for subpath in ['sources/', 'sources/expRNN/', 'source/nnRNN/']]
for path in import_path:
  sys.path.append(path)

from data_module import PTBDataModule
from trivializations import cayley_map, expm
from initialization import henaff_init_, cayley_init_
from data_module import PTBDataModule
from torch.nn import Embedding
from model_loader import Model
from custom_modules import asRNN
from orthogonal import modrelu

parser = argparse.ArgumentParser(description='PTB-c')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--rmsprop_lr', type=float, default=1e-3)
parser.add_argument('--rmsprop_constr_lr', type=float, default=1e-4)
parser.add_argument('--alpha', type=float, default=0.99)
parser.add_argument('--clip_norm', type=float, default=-1) #set negative to disable
parser.add_argument("-m", "--mode",
                    choices=["exprnn", "dtriv", "cayley", "lstm", "rnn"],
                    default="exprnn",
                    type=str)            
parser.add_argument("--nonlinear",
                    choices=["asrnn", "modrelu"],
                    default="asrnn",
                    type=str)
parser.add_argument("-a", type=float, default=8e-1)
parser.add_argument("-b", type=float, default=3)
parser.add_argument("--eps", type=float, default=0)
parser.add_argument('--K', type=str, default="100", help='The K parameter in the dtriv algorithm. It should be a positive integer or "infty".')
parser.add_argument("--init",
                    choices=["cayley", "henaff"],
                    default="cayley",
                    type=str)
parser.add_argument('--bptt', type=int, default=150)
parser.add_argument('--log-interval', type=int, default=32, metavar='N',
                    help='report interval')
parser.add_argument('--rho_rat_den', type=int, default=10)
parser.add_argument('--forget_bias', type=int, default=1)
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
#Setting
args = parser.parse_args(sys.argv[1:])
print(sys.argv, args)

epochs      = args.epochs
batch_size  = args.batch_size
eval_batch_size = 10
batch_first = False
many_to_many = True
datamodule = PTBDataModule(dataset_path, batch_size, eval_batch_size, args.bptt, device)

input_size = args.emsize
hidden_size = args.hidden_size
output_size = datamodule.output_size

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

if args.emsize > 0:
  embed_layer = Embedding(datamodule.input_size, args.emsize) # Token2Embeddings
  input_size = args.emsize
else:
  embed_layer = None
  input_size = datamodule.input_size

#Initialize Model
model = Model(input_size, hidden_size, datamodule.output_size, nonlinearity, initializer_skew = init,
              mode = mode, param = param, args=args, embed_layer=embed_layer, batch_first=batch_first).to(device)
initrange = 0.1
model.embed_layer.weight.data.uniform_(-initrange, initrange)
model.lin.bias.data.fill_(0)
model.lin.weight.data.uniform_(-initrange, initrange)

#Initialize Optimizers
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

column_names = ['epoch', 'train_bpc', 'valid_bpc', 'valid_acc', 'train_time']
train_log = pd.DataFrame(columns = column_names)

import math
def train(model, datamodule):
  global args
  bpcs = []
  total_loss = 0
  correct = 0
  processed = 0
  embed_grad_norms = []
  hidden = model.init_state(batch_size)
  data_source = datamodule.train
  model.train()
  for batch, i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
    data, targets = datamodule.get_batch(data_source,i)
    hidden = model.detach_state(hidden)
    #Zero-ing grad for gradient descent
    for optim in model.optim_list:
        if optim:
            optim.zero_grad()
    output, hidden = model(data, hidden)
    output = output.view(-1, datamodule.output_size)
    loss = torch.nn.functional.cross_entropy(output, targets)
    #Gradient descent
    loss.backward()
    if args.clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    for optim in model.optim_list:
        if optim:
            optim.step()
    total_loss += loss.item()
    if batch % args.log_interval == 0 and batch > 0:
      bpcs.append(total_loss / args.log_interval /math.log(2))
      print('train: epoch {}| batch {}/{} ({:.3f}%)| bpc {:.3f}|'.format(epoch+1, batch, (len(datamodule.train) // args.bptt), 100 * batch/(len(datamodule.train) // args.bptt), bpcs[-1]))
      total_loss = 0
    if exit_signal: #Safe interruption to avoid GPU memory buffer overflow
        break
        
  return np.mean(bpcs)

def evaluate(model, datamodule, mode):
    # Turn on evaluation mode which disables dropout.
    global args
    model.eval()
    total_loss = 0
    correct = 0
    processed = 0
    data_source = eval("datamodule." + mode)
    hidden = model.init_state(10)
    with torch.no_grad():
      for i in range(0, data_source.size(0) - 1, args.bptt):
          data, targets = datamodule.get_batch(data_source, i)
          output, hidden = model(data, hidden)
          output_flat = output.view(-1, datamodule.output_size)
          total_loss += len(data) * torch.nn.functional.cross_entropy(output_flat, targets).item()
          correct += torch.eq(torch.argmax(output_flat,dim=1),targets).sum().item()
          processed += targets.shape[0]
          
    return total_loss / len(data_source) / math.log(2), 100 * correct/processed

scheduler_list = [torch.optim.lr_scheduler.StepLR(optim,1,gamma=0.5, verbose = True) for optim in model.optim_list]

best_valid_bpc = float('inf')
for epoch in range(epochs):
  start = time.time()
  train_bpc = train(model, datamodule)
  traintime = time.time()-start
  
  valid_bpc, valid_acc = evaluate(model, datamodule, mode = "valid")
  
  if valid_bpc < best_valid_bpc:
    best_valid_bpc = valid_bpc
    model.save_state(expr_path+'/best.ckpt')
  else:
    for scheduler in scheduler_list:        
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        scheduler.step()
      
  print('=' * 89)
  print('train: epoch {}/{}| bpc {:.3f}| runtime {:.3f}|'.format(epoch+1, epochs, train_bpc, traintime))
  print('valid: epoch {}/{}| best bpc {:.3f}| bpc {:.3f}| acc {:.2f}%|'.format(epoch+1, epochs, best_valid_bpc, valid_bpc, valid_acc))
  print('=' * 89)

  new_row = pd.Series({"epoch": epoch, "train_bpc": train_bpc, "valid_bpc": valid_bpc, 'valid_acc': valid_acc, 'train_time':traintime})
  train_log = model.update_log_df(new_row, train_log, expr_path+'/train_log.pkl')
  if exit_signal:
    print('-' * 89)
    print('Exiting from training early')
    break

model.load_state(expr_path+'/best.ckpt')
test_bpc, test_acc = evaluate(model, datamodule, mode = "test")
print('=' * 89)
print('test: bpc {}| acc {}%|'.format(test_bpc, test_acc))
print('=' * 89)