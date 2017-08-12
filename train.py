import argparse
import codecs
import string
import random

import torch
from torch.autograd import Variable

from model import Model


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
