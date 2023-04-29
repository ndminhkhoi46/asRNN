#=============================
#Import
#=============================
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.utils.data as data
import torch
#from pytorch_lightning import LightningDataModule
import random
import numpy as np
import os
import sys
import requests
import tarfile

#-----------------------------MNIST data.Dataset-----------------------------#
class MNISTDataModule:
    def __init__(self, sequence_length, input_size, batch_size, path, permute_seed):        
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = 10
        self.batch_size = batch_size
        self.path = path

        if permute_seed:
            rng_permute = np.random.RandomState(permute_seed)
            input_permutation = torch.from_numpy(rng_permute.permutation(784))      
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1)[input_permutation].view(1, self.sequence_length, self.input_size))])
        else:
            transform = transforms.ToTensor()

        mnist_train_data = MNIST(root=self.path, train=True, download=True, transform=transform)
        mnist_test_data = MNIST(root=self.path, train=False, download=False, transform=transform)                                   
        self.train_len = mnist_train_data.__len__()                                 
        self.test_len = mnist_test_data.__len__()
        #https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader
        self.mnist_train_loader = data.DataLoader(mnist_train_data,batch_size=self.batch_size, shuffle = True, generator = torch.Generator(), pin_memory = True)
        self.mnist_test_loader = data.DataLoader(mnist_test_data,batch_size=1000, shuffle = False, pin_memory = True)                                                                                    
    def enum(self, train=True):
        if train:
            return self.mnist_train_loader
        else:
            return self.mnist_test_loader
    def len(self, train=True):
        if train:
            return self.train_len
        else:
            return self.test_len
#-----------------------------Copy Memory Task data.Dataset-----------------------------#
class CopyMemoryDataModule:
    def __init__(self, recall_length, delay_length, iterations):
        self.recall_length = recall_length
        self.delay_length = delay_length
     
        self.sequence_length = self.delay_length + 2 * self.recall_length
        self.input_size = 10#2-8 digits, blank letter 0, marker 9
        self.output_size = 9#2-8 digits, blank letter 0
    def copying_data(self, L, K, batch_size):
        seq = np.random.randint(1, high=9, size=(batch_size, K))
        zeros1 = np.zeros((batch_size, L))
        zeros2 = np.zeros((batch_size, K-1))
        zeros3 = np.zeros((batch_size, K+L))
        marker = 9 * np.ones((batch_size, 1))

        x = torch.LongTensor(np.concatenate((seq, zeros1, marker, zeros2), axis=1))
        y = torch.LongTensor(np.concatenate((zeros3, seq), axis=1))
        return x, y
        
    def onehot(self, out, input):
        out.zero_()
        in_unsq = torch.unsqueeze(input, 2)
        out.scatter_(2, in_unsq, 1)
#-----------------------------PTB data.Dataset-----------------------------#
#=============================
#Dictionary
#=============================
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
#=============================
#Data-module
#=============================
class PTBDataModule:
    def __init__(self, path, batch_size, eval_batch_size, bptt, device):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path,'ptb.char.train.txt'))
        self.train = self.batchify(self.train, batch_size)
        self.valid = self.tokenize(os.path.join(path,'ptb.char.valid.txt'))
        self.valid = self.batchify(self.valid, eval_batch_size)
        self.test = self.tokenize(os.path.join(path,'ptb.char.test.txt'))
        self.test = self.batchify(self.test, eval_batch_size)
        
        self.sequence_length = bptt
        self.input_size = len(self.dictionary)
        self.output_size = len(self.dictionary)
        
        if device:
            self.train = self.train.to(device)
            self.valid = self.valid.to(device)
            self.test = self.test.to(device)
            
        self.bptt = bptt
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids
    def get_batch(self, source, i):
        # source: size(total_len//bsz, bsz)
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target
    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the data.Dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous() #Remark: There is a transposing operation here
        return data
