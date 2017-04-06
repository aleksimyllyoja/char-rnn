# Recurrent Neural Network (RNN) model

+ Character-level language model implementation of Karpathy's char-rnn(http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
with Long Short Term Memory (LSTM) shell.

Implemented with powerful yet flexible neural network's framework [Chainer](https://github.com/pfnet/chainer).


## Requirement

+ Python (>= 3.4.0) recommended.


## Set-up

+ Install required packages:

  + Traditional way:

  ```
  $ pip3 install -r requirements.txt
  ```

  + `virtualenv` users:

  ```
  $ virtualenv -p /usr/bin/python3 venv
  $ source venv/bin/activate
  $ pip install -r requirements.txt
  ```

## Demo

+ [Training](#training) (*Skip if you want to see directly generating process with pre-trained model*)

```
$ python train.py --data data/harry.txt
```

An example input `harry.txt` already supported with text format of J.K.Rowling's `Harry Potter and The Order of The Phoenix`


+ [Generating](#generating)

```
$ python gen.py --model result/model_iter_65133 \
  --vocab result/vocab.bin \
  --pretext 'Harry Potter'
```

Go on and try it with an example of trained data from `harry.txt`


## Training

+ Basic:

```
$ python train.py --data data/input.txt
```

+ Full options:

```
$ python train.py --data data/harry.txt \
  --result_dir result \
  --n_units 128 \
  --n_epochs 30 \
  --resume result/model_iter_{n}
```

Result will be stored in `result` directory:

+ `model_iter_{n}`: Serialized model object can be used for resume training or predicted [generating](#generating)

+ `vocab.bin`: Binary object contains vocabulary of training object

## Generating

+ Basic

```
$ python gen.py --model result/model_iter_{n} \
  --vocab result/vocab.bin \
  --pretext 'Hello'
```

+ Full options:

```
$ python gen.py --model result/model_iter_{n} \
  --vocab result/harry_potter.bin \
  --pretext 'Harry' \
  --length 2000 \
  --n_units 128
```

Result is putted into `stdout`, you can redirect it to a file with a simple trick:

```
$ python gen.py --model result/model_iter_{n} > sample.txt
```

## References

Code is heavily inspired from these sources:

+ First char-rnn implementation written with Chainer https://github.com/yusuketomoto/chainer-char-rnn

+ Chainer's PTB example https://github.com/pfnet/chainer/tree/master/examples/ptb
