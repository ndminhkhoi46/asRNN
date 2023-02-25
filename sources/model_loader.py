import torch
import torch.nn as nn
from expRNN.orthogonal import OrthogonalRNN
from custom_modules import RNNCell

import os
import random
import numpy as np
import pandas as pd

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nonlinearity, initializer_skew, mode, param, args, embed_layer, batch_first):
        super(Model, self).__init__() 
        self.hidden_size = hidden_size
        if batch_first:
            self.dim_T = 1
        else:
            self.dim_T = 0
        self.args = args
        
        has_bias = not ("modrelu" in nonlinearity.__str__())
        if args.mode == "lstm":
            self.rnn = nn.LSTMCell(input_size, hidden_size)
            #https://github.com/pytorch/pytorch/issues/750
            #https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
            self.rnn.bias_ih[hidden_size:2*hidden_size].data.fill_(self.args.forget_bias/2)
            self.rnn.bias_hh[hidden_size:2*hidden_size].data.fill_(self.args.forget_bias/2)
        elif args.mode == "rnn":
            self.rnn = RNNCell(input_size, hidden_size, nonlinearity, has_bias, initializer_skew)
        else:
            if args.mode == "cayley":
                rho = hidden_size // args.rho_rat_den
                self.register_buffer('D', torch.diag(torch.cat([torch.ones(hidden_size - rho), -torch.ones(rho)], dim = 0)))
            self.rnn = OrthogonalRNN(input_size, hidden_size, nonlinearity=nonlinearity, has_bias=has_bias, initializer_skew=initializer_skew, mode=mode, param=param)
            
        self.embed_layer = embed_layer
        self.lin = nn.Linear(hidden_size, output_size, bias=True)
        self.loss = nn.CrossEntropyLoss()
        
        self.optim_list = None
    def init_state(self, batch_size):
        device = next(self.parameters()).device
        if self.args.mode == "lstm":
            state = (torch.zeros((batch_size, self.hidden_size), device=device),
                    torch.zeros((batch_size, self.hidden_size), device=device))
        else:
            state = torch.zeros((batch_size, self.hidden_size), device=device)
        return state
    def detach_state(self, state):
        if self.args.mode == "lstm":
            return tuple(v.clone().detach() for v in state)
        else:
            return state.detach()
    def forward(self, data, hidden): 
        if not self.args.mode == "lstm":
            if hasattr(self.rnn.nonlinearity, 'reset_cache'):
                self.rnn.nonlinearity.reset_cache()  
        hs = []    
        if self.embed_layer:
            x = self.embed_layer(data)
        else:
            x = data
        for t in range(x.shape[self.dim_T]):
            if self.args.mode == "cayley":
                hidden = hidden.mm(self.D)
            out_rnn, hidden = self.rnn(torch.select(x, self.dim_T, t), hidden)
            hs.append(hidden)
            if self.args.mode == "lstm":
                hidden = (out_rnn, hidden)
        output = torch.stack(hs, dim = self.dim_T)
        decoded = self.lin(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
    def correct(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=-1), y).float().sum()
    def update_log_df(self, new_row, df, df_path):
        idx_col_name = new_row.index[0]
        df = pd.concat([df, new_row.to_frame().T])
        df.to_pickle(df_path)
        return df
    def save_state(self, path):
        torch.save(self.state_dict(), path)
    def load_state(self, path):
        self.load_state_dict(torch.load(path))