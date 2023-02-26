import signal

exit_signal = False
def handler(signum, frame):
    global exit_signal
    exit_signal = True   
signal.signal(signal.SIGINT, handler) 
import os
import sys
import argparse
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import random
import os
import time
import torch.nn as nn
import subprocess
from scipy import stats

def install(name):
    subprocess.call(['pip', 'install', name])
install('scipy')
install('pandas')

root_path = './'
expr_path = root_path + 'Benchmark/Copy Task/Recall 10, Delay 1000/asRNN(0,0,1e-5)'
os.makedirs(expr_path, exist_ok=True)
import_path = [root_path + subpath for subpath in ['sources/', 'sources/expRNN/']]
for path in import_path:
  sys.path.append(path)

from model_loader import Model
from custom_modules import asRNN
from orthogonal import modrelu
from trivializations import cayley_map, expm
from initialization import henaff_init_, cayley_init_
from data_module import CopyMemoryDataModule
from torch.nn import Embedding
import pandas as pd

sys.argv = ['']
parser = argparse.ArgumentParser(description='Copying Memory Task')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=138)
parser.add_argument('--iterations', type=int, default=4000)
parser.add_argument('--rmsprop_lr', type=float, default=1e-3)
parser.add_argument('--rmsprop_constr_lr', type=float, default=1e-4)
parser.add_argument('--alpha', type=float, default=0.99)
parser.add_argument('--clip_norm', type=float, default=10) #negative to disable
parser.add_argument("-m", "--mode",
                    choices=["exprnn", "dtriv", "cayley", "lstm", "rnn"],
                    default="exprnn",
                    type=str)
parser.add_argument("--nonlinear",
                    choices=["asrnn", "modrelu"],
                    default="asrnn",
                    type=str)
parser.add_argument("-a", type=float, default=0)
parser.add_argument("-b", type=float, default=0)
parser.add_argument("--eps", type=float, default=1e-5)
parser.add_argument('--K', type=str, default="100", help='The K parameter in the dtriv algorithm. It should be a positive integer or "infty".') #default 100
parser.add_argument("--init",
                    choices=["cayley", "henaff"],
                    default="henaff",
                    type=str)
parser.add_argument('--random-seed', type=int, default=5544,
                    help='random seed')
parser.add_argument('--recall_length', type=int, default=10)
parser.add_argument('--delay_length', type=int, default=1000)
parser.add_argument('--rho_rat_den', type=int, default=2)
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

#Setting up data module
# Load data   
batch_size  = args.batch_size
datamodule = CopyMemoryDataModule(args.recall_length, args.delay_length, batch_size)
input_size = datamodule.input_size
hidden_size = args.hidden_size
iterations      = args.iterations

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
    

#Initialize Model
model = Model(input_size, hidden_size, datamodule.output_size, nonlinearity, initializer_skew = init,
              mode = mode, param = param, args=args, embed_layer=None, batch_first=True).to(device)
model.lin.bias.data = torch.zeros_like(model.lin.bias.data)

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

column_names = ['step', 'train_loss', 'train_acc', 'train_time']
train_log = pd.DataFrame(columns = column_names)

def pred(model, datamodule, train = True):
  global device
  batch_loss = 0.0
  batch_acc = 0.0

  batch_x, batch_y = datamodule.enum()
  batch_x, batch_y = batch_x.to(device).view(-1, datamodule.sequence_length, datamodule.input_size), batch_y.to(device)
  state = model.init_state(batch_size)
  logits, _ = model(batch_x, state)
  output = logits.view(-1, datamodule.output_size)
  loss = model.loss(output, batch_y)
  if train:
    model.zero_grad() 
    loss.backward()

    if model.args.clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.args.clip_norm)
    for optim in model.optim_list:
        if optim:
            optim.step()
  batch_loss = loss.item()
  batch_correct = model.correct(output, batch_y).item()
  batch_acc = batch_correct / batch_y.size(0)
  batch_acc *= 100
  
  return batch_loss, batch_acc
model.train()
best_train_loss = float('inf')
for step in range(iterations):
    start = time.time()
    train_loss, train_acc = pred(model, datamodule, train = True)
    traintime = time.time()-start
    print('(train): step {}| loss {:.3f}| acc {:.2f}%| best_loss {}| train time: {:.2f}'.format(step, train_loss, train_acc, best_train_loss, traintime))
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        model.save_state(expr_path+'/best.ckpt')
    new_row = pd.Series({"step": step, "train_loss": train_loss, "train_acc": train_acc,
    'train_time': traintime})
    train_log = model.update_log_df(new_row, train_log, expr_path+'/train_log.pkl')  
    if exit_signal:
        print('-' * 89)
        print('Exiting from training early')
        break
          
print('-' * 89)
model.load_state(expr_path+'/best.ckpt')
model.eval()
with torch.no_grad():
    test_loss, test_acc = pred(model, datamodule, train = False)
    print('(test): loss {}| acc {}%'.format(test_loss, test_acc))

#Ring a bell to notice the completion
print('\007')
