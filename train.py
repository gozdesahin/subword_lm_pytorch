#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Clara Vania
import numpy as np
import argparse
import time
import os
import pickle
import sys
from utils import TextLoader,get_last_model_path
from model import *
from torch.autograd import Variable
from optimizer import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/train.txt',
                        help="training data")
    parser.add_argument('--dev_file', type=str, default='data/dev.txt',
                        help="development data")
    parser.add_argument('--output_vocab_file', type=str, default='',
                        help="vocab file, only use this if you want to specify special output vocabulary!")
    parser.add_argument('--output', '-o', type=str, default='train.log',
                        help='output file')
    parser.add_argument('--save_dir', type=str, default='model',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=200,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--unit', type=str, default='char-ngram',
                        help='char, char-ngram, morpheme, word, or oracle')
    parser.add_argument('--composition', type=str, default='addition',
                        help='none(word), addition, or bi-lstm')
    parser.add_argument('--lowercase', dest='lowercase', action='store_true',
                        help='lowercase data', default=False)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--out_vocab_size', type=int, default=5000,
                        help='size of output vocabulary')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--patience', type=int, default=3,
                        help='the number of iterations allowed before decaying the '
                             'learning rate if there is no improvement on dev set')
    parser.add_argument('--validation_interval', type=int, default=1,
                        help='validation interval')
    parser.add_argument('--init_scale', type=float, default=0.1,
                        help='initial weight scale')
    parser.add_argument('--param_init_type', type=str, default="uniform",
                        help="""Options are [orthogonal|uniform|xavier_n|xavier_u]""")
    parser.add_argument('--grad_clip', type=float, default=2.0,
                        help='maximum permissible norm of the gradient')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='the decay of the learning rate')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='the probability of keeping weights in the dropout layer')
    parser.add_argument('--gpu', type=int, default=0,
                        help='the gpu id, if have more than one gpu')
    parser.add_argument('--optimization', type=str, default='sgd',
                        help='sgd, momentum, or adagrad')
    parser.add_argument('--train', type=str, default='softmax',
                        help='sgd, momentum, or adagrad')
    parser.add_argument('--SOS', type=str, default='false',
                        help='start of sentence symbol')
    parser.add_argument('--EOS', type=str, default='true',
                        help='end of sentence symbol')
    parser.add_argument('--ngram', type=int, default=3,
                        help='length of character ngram (for char-ngram model only)')
    parser.add_argument('--char_dim', type=int, default=200,
                        help='dimension of char embedding (for C2W model only)')
    parser.add_argument('--morph_dim', type=int, default=200,
                        help='dimension of morpheme embedding (for M2W model only)')
    parser.add_argument('--word_dim', type=int, default=200,
                        help='dimension of word embedding (for C2W model only)')
    parser.add_argument('--cont', type=str, default='false',
                        help='continue training')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for random initialization')
    parser.add_argument('--lang', type=str, default='tr',
                        help='Language (others|tr)')
    args = parser.parse_args()
    # check cuda
    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    otype = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    args.dtype = dtype
    args.otype = otype
    args.use_cuda = use_cuda
    train(args)

def lossCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    crit = nn.NLLLoss(weight, size_average=False)
    return crit

def run_epoch(m, data, data_loader, optimizer, eval=False):
    if eval:
        m.eval()
    else:
        m.train()
    # epoch size, training datayi bir kere process etmesi icin gereken step sayisi
    # zero division error for small datasets
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    crit = lossCriterion(data_loader.out_vocab_size)
    m.lm_hidden = m.init_hidden(m.num_layers,numdirec=1, batchsize=m.batch_size)
    if data_loader.composition == "bi-lstm":
        m.comp_hidden = m.init_hidden(1, numdirec=2, batchsize=m.batch_size)
    for step, (x, y) in enumerate(data_loader.data_iterator(data, m.batch_size, m.num_steps)):
        # make them matrix
        # x can be values, x can be indices
        if (data_loader.composition=="bi-lstm") or \
                (data_loader.composition == "none"): # indices are returned
            x = torch.LongTensor(x).type(m.otype)
        elif data_loader.composition=="addition": # values are returned
            x = torch.FloatTensor(x).type(m.dtype)
        # y is always indices
        y = torch.LongTensor(y).type(m.otype)
        # move input tensors to gpu if possible
        if m.use_cuda:
            x = x.cuda()
            y = y.cuda()
            crit = crit.cuda()
        # require_grad by default false
        x_var = Variable(x, volatile=eval)
        y_var = Variable(y, volatile=eval)
        # zero the gradients
        m.zero_grad()
        # Shall I initialize them with zero after each sequence of length number of steps - (e.g. 20)
        # Or shall I initialize them with their previous value by detaching them from their history
        # tensorflow code initialized with previous state and they don't have the history problem
        m.lm_hidden = repackage_hidden(m.lm_hidden)
        if data_loader.composition=="bi-lstm":
            m.comp_hidden = repackage_hidden(m.comp_hidden)
        #m.lm_hidden = m.init_hidden(m.num_layers)
        log_probs = m(x_var)
        training_labels = y_var.view(log_probs.size(0))
        loss = crit(log_probs, training_labels).div(m.batch_size)
        # stupid vector -> value transformations, waste of time!
        costs += loss.data[0]
        # in tutorial they multiply it by the number of steps ?
        # costs += loss.data[0] * model.num_steps
        # however I give size_average=False parameter, so not sure ?
        iters += m.num_steps
        if not eval:
            # go backwards and update weights
            loss.backward()
            optimizer.step()
            # report
            if not eval and step % (epoch_size // 10) == 10:
                print("perplexity: %.3f speed: %.0f wps" %
                      ( np.exp(costs / iters),
                       iters * m.batch_size / (time.time() - start_time)))

    # calculate perplexity
    # this is cost per word
    cost_norm = (costs/iters)
    ppl = math.exp(min(cost_norm, 100.0))
    return ppl

def train(args):
    start = time.time()
    save_dir = args.save_dir
    try:
        os.stat(save_dir)
    except:
        os.mkdir(save_dir)

    args.eos = ''
    args.sos = ''
    if args.EOS == "true":
        args.eos = '</s>'
        args.out_vocab_size += 1
    if args.SOS == "true":
        args.sos = '<s>'
        args.out_vocab_size += 1

    local_test = False
    if local_test:
        # Gozde
        # char, char-ngram, morpheme, word, or oracle
        args.unit = "oracle"
        args.composition = "bi-lstm"
        args.train_file = "data/train.morph"
        args.dev_file = "data/dev.morph"
        args.batch_size = 12
        # End of test

    data_loader = TextLoader(args)
    train_data = data_loader.train_data
    dev_data = data_loader.dev_data

    fout = open(os.path.join(args.save_dir, args.output), "a")

    args.word_vocab_size = data_loader.word_vocab_size

    if args.unit != "word":
        args.subword_vocab_size = data_loader.subword_vocab_size
    fout.write(str(args) + "\n")

    # Statistics of words
    fout.write("Word vocab size: " + str(data_loader.word_vocab_size) + "\n")

    # Statistics of sub units
    if args.unit != "word":
        fout.write("Subword vocab size: " + str(data_loader.subword_vocab_size) + "\n")
        if args.composition == "bi-lstm":
            if args.unit == "char":
                fout.write("Maximum word length: " + str(data_loader.max_word_len) + "\n")
                args.bilstm_num_steps = data_loader.max_word_len
            elif args.unit == "char-ngram":
                fout.write("Maximum ngram per word: " + str(data_loader.max_ngram_per_word) + "\n")
                args.bilstm_num_steps = data_loader.max_ngram_per_word
            elif args.unit == "morpheme" or args.unit == "oracle":
                fout.write("Maximum morpheme per word: " + str(data_loader.max_morph_per_word) + "\n")
                args.bilstm_num_steps = data_loader.max_morph_per_word
            else:
                sys.exit("Wrong unit.")
        elif args.composition == "addition":
            if args.unit not in ["char-ngram", "morpheme", "oracle"]:
                sys.exit("Wrong composition.")
        else:
            sys.exit("Wrong unit/composition.")
    else:
        if args.composition != "none":
            sys.exit("Wrong composition.")

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    print(args)
    if args.unit == "word":
        lm_model = WordModel
    elif args.composition == "addition":
        lm_model = AdditiveModel
    elif args.composition == "bi-lstm":
        lm_model = BiLSTMModel
    else:
        sys.exit("Unknown unit or composition.")

    print("Begin training...")

    mtrain = lm_model(args)
    if args.use_cuda:
        mtrain = mtrain.cuda()

    nParams = sum([p.nelement() for p in mtrain.parameters()])
    print('* number of parameters: %d' % nParams)

    optim = Optim(
        args.optimization, args.learning_rate, args.grad_clip,
        lr_decay=args.decay_rate,
        patience=args.patience
    )
    # update all parameters
    optim.set_parameters(mtrain.parameters())

    dev_pp = 10000000.0

    if args.cont == 'true':  # continue training from a saved model
        # get model parameters
        model_path, e = get_last_model_path(args.save_dir)
        saved_model = torch.load(model_path)
        mtrain.load_state_dict(saved_model['state_dict'])
        # get optimizer states
        # not saving learning rate (probably too small so it won't continue training)
        optim.last_ppl = saved_model['last_ppl']
    else:
        # process each epoch
        e = 1

    while e <= args.num_epochs:
        print("Epoch: %d" % e)
        print("Learning rate: %.3f" % optim.lr)

        #  (1) train for one epoch on the training set
        train_perplexity = run_epoch(mtrain, train_data, data_loader,optim, eval=False)
        print("Train Perplexity: %.3f" % train_perplexity)

        #  (2) evaluate on the validation set
        dev_perplexity = run_epoch(mtrain, dev_data, data_loader, optim, eval=True)
        print("Valid Perplexity: %.3f" % dev_perplexity)

        #  (3) update the learning rate
        optim.updateLearningRate(dev_perplexity, e)

        # (4) save results and report
        diff = dev_pp - dev_perplexity
        if diff >= 0.1:
            print("Achieve highest perplexity on dev set, save model.")
            checkpoint = {
                'state_dict': mtrain.state_dict(),
                'last_ppl':optim.last_ppl
            }
            torch.save(checkpoint,
                       '%s/%s-%d.pt' % (save_dir, "model", e))
            dev_pp = dev_perplexity

        # write results to file
        fout.write("Epoch: %d\n" % e)
        fout.write("Learning rate: %.3f\n" % optim.lr)
        fout.write("Train Perplexity: %.3f\n" % train_perplexity)
        fout.write("Valid Perplexity: %.3f\n" % dev_perplexity)
        fout.flush()

        if optim.lr < 0.0001:
            print('Learning rate too small, stop training.')
            break

        e += 1

    print("Training time: %.0f" % (time.time() - start))
    fout.write("Training time: %.0f\n" % (time.time() - start))

if __name__ == '__main__':
    main()