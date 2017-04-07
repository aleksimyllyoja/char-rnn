from __future__ import division
from __future__ import print_function

import argparse
import os
import codecs
import _pickle
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from char_rnn import charRNN


def load_data(data_path):
    """Process data into 3 informations:
    + dataset: Numpy arrary of integers, contains ID of the word in vocab
    + words: List of character-level words
    + vocab: Vocabulary of all words exists in data
    """

    vocab = {}
    print("---> Preparing data from %s" % data_path)
    words = codecs.open(data_path, 'rb', 'utf-8').read()
    words = list(words)
    dataset = np.ndarray((len(words),), dtype=np.int32)

    vocab_len = 0
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = vocab_len
            vocab_len += 1

        dataset[i] = vocab[word]

    print("+ Corpus length: %d" % len(words))
    print("+ Vocab size: %d" % len(vocab))

    return dataset, words, vocab


class ParalellSequentialIterator(chainer.dataset.Iterator):
    """This iterattor returns a pair of current words and the next words
    Each example is a part of sequences starting from the different offsets
    equally spaced within the whole sequences
    """

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size

        # Number of completed sweeps over the dataset
        # In this case, this is incremented if everyword
        # is visited at least once after the last increment
        self.epoch = 0
        # True if the epoch is incremented at the last iteration
        self.is_new_epoch = False
        self.repeat = repeat

        length = len(dataset)
        # Offsets maitain the position of each sequence in the mini-batch
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        # Count of each forward + backward pass of batch_size (__next__)
        self.iteration = 0

    def __next__(self):
        # Return a list representing a mini-batch
        # Each item is represented by a pair of 2 word IDs.
        # [current position, next position]
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration

        cur_words = self.get_words()
        # move to next offset to get list of next words
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = (self.epoch < epoch)
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        # List of current words
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


class BPTTUpdater(training.StandardUpdater):
    """Custom Updater for truncated BackProp Through Time (BPTT)
    """

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device
        )
        self.bprop_len = bprop_len

    def update_core(self):
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of 2 word IDs)
            batch= train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to device
            x, t = self.converter(batch, self.device)

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='data/input.txt')
    parser.add_argument('--vocab', '-v', type=str, default='vocab.bin')
    parser.add_argument('--n_units', '-n', type=int, default=128)
    parser.add_argument('--n_epochs', '-e', type=int, default=30)
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--result_dir', '-o', type=str, default='result')
    args = parser.parse_args()

    # Hard-coded setup
    batch_size = 20

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    # Load train data
    train, words, vocab = load_data(args.data)
    train_len = train.shape[0]

    # Dump vocab to a usable binary object
    _pickle.dump(vocab, open('%s/vocab.bin' % args.result_dir, 'wb'))

    # Iterate for creating a  batch of sequences at different positions
    train_iter = ParalellSequentialIterator(train, batch_size)

    # Pick the model
    rnn = charRNN(len(vocab), args.n_units)
    model = L.Classifier(rnn)

    # Intialize gpu 0
    chainer.cuda.get_device(0).use()
    model.to_gpu()

    # Optimizer setup
    optimizer = chainer.optimizers.RMSprop(lr=2e-3, alpha=0.95, eps=1e-8)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    # Setup a trainer
    bprop_len = 35 # Number of words in each mini-batch
    updater = BPTTUpdater(train_iter, optimizer, bprop_len, 0)
    trainer = training.Trainer(updater, (args.n_epochs, 'epoch'), out='result')

    # Some support extensions
    interval = 500
    trainer.extend(extensions.LogReport(trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration']
    ))
    trainer.extend(extensions.ProgressBar(
        update_interval=10
    ))
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'
    ))

    # Resume from model snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Begin training
    trainer.run()


if __name__ == '__main__':
    main()
