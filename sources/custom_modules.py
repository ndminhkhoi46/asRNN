import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from expRNN.initialization import henaff_init_, cayley_init_
from expRNN.orthogonal import Parametrization, Orthogonal, modrelu
from expRNN.trivializations import expm
import math
#-----------------------------RNN Module-----------------------------#
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity, has_bias, skew_init):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_kernel = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=has_bias)
        self.recurrent_kernel = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.nonlinearity = nonlinearity
        self.reset_parameters(skew_init)

    def reset_parameters(self, skew_init):
        with torch.no_grad():
            self.recurrent_kernel.weight.data = skew_init(self.recurrent_kernel.weight.data)
            self.recurrent_kernel.weight.data = torch.matrix_exp(self.recurrent_kernel.weight.data)
            if "modrelu" in self.nonlinearity.__str__():
                self.input_kernel.weight.data = nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")
            elif "AMSAF_SVD" in self.nonlinearity.__str__():
                self.input_kernel.bias.data = torch.zeros_like(self.input_kernel.bias.data)
                self.input_kernel.weight.data = nn.init.orthogonal_(self.input_kernel.weight.data)

    def default_hidden(self, input):
        return input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        out = self.nonlinearity(out)
        
        return out, out
          
class AMSAF_SVD(nn.Module):
    def __init__(self, size, initializer_skew, mode, param, s_min, s_max, s_epsilon):
        super(AMSAF_SVD, self).__init__()
        self.s_epsilon = s_epsilon
        #Declaration
        self.S = nn.Parameter(torch.empty(size))
        self.U = Orthogonal(size, size, initializer_skew, mode, param)
        #Initialization
        with torch.no_grad():
            self.S.data = nn.init.uniform_(self.S.data, a=s_min, b=s_max)
            self.U.A.data = torch.zeros_like(self.U.A.data)
            self.W = None
            self.W_inv = None
    def reset_cache(self):
        S = self.S.abs()+self.s_epsilon
        self.W = self.U.B.mm(S.diag())
        self.W_inv = (1/S).diag().mm(self.U.B.t())
    def forward(self, inputs):
        enc = inputs.mm(self.W.t())
        dec = (enc.tanh()).mm(self.W_inv.t())
        return dec