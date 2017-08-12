import argparse
import codecs
import string
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import RNN


# Data
all_characters = string.printable
n_characters = len(all_characters)
source = codecs.open("harry.txt", "r", encoding="utf-8", errors="ignore").read()
source_len = len(source)
chunk_len = 200

# Hyper parameters
n_hidden = 50
n_layers = 2


def char_tensor(string, gpu=-1):
    """Turn string into list of longs"""
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c]  = all_characters.index(string[c])

    if gpu >= 0:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def random_chunk():
    start_index = random.randint(0, source_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return source[start_index:end_index]

def random_training_set(gpu=-1):
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1], gpu)
    target = char_tensor(chunk[1:], gpu)

    return inp, target


class Model():
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, gpu=-1):
        self.decoder = RNN(input_size, hidden_size, output_size, n_layers, gpu)
        if gpu >= 0:
            print("Use GPU %d" % torch.cuda.current_device())
            self.decoder.cuda()

        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, inp, target, chunk_len=200):
        hidden = self.decoder.init_hidden()
        self.decoder.zero_grad()
        loss = 0

        for c in range(chunk_len):
            out, hidden = self.decoder(inp[c], hidden)
            loss += self.criterion(out, target[c])

        loss.backward()
        self.optimizer.step()

        return loss.data[0] / chunk_len

    def evaluate(prime_input, predict_len=100, temperature=0.8):
        hidden = self.decoder.init_hidden()

        # Use prime string to build up hidden state
        # for p in range(len() - 1):



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
    argparser.add_argument('--frequency', type=int, default=10, help='Frequently check loss with interval')
    argparser.add_argument('--gpu', type=int, default=-1, help='Id of GPU (-1 indicates CPU)')
    args = argparser.parse_args()

    # Build model
    model = Model(n_characters, n_hidden, n_characters, n_layers, args.gpu)

    # Train
    print("Train with %d epochs" % args.epoch)

    for e in range(args.epoch):
        loss = model.train(*random_training_set(args.gpu))

        if (e+1) % args.frequency == 0:
            print("[EPOCH %d] loss %.4f" % (e+1, loss))
