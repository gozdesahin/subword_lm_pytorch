import argparse
import os, sys
import pickle
from utils import TextLoader,get_last_model_path
from model import *
from torch.autograd import Variable
from optimizer import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/test.txt',
                        help="test file")
    parser.add_argument('--save_dir', type=str, default='model',
                        help='directory of the checkpointed models')
    args = parser.parse_args()
    test(args)


def lossCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    crit = nn.NLLLoss(weight, size_average=False)
    return crit

def run_epoch(m, data, data_loader, eval=True):
    m.eval()
    costs = 0.0
    iters = 0
    crit = lossCriterion(data_loader.out_vocab_size)
    m.lm_hidden = m.init_hidden(m.num_layers)
    for step, (x, y) in enumerate(data_loader.data_iterator(data, m.batch_size, m.num_steps)):
        x = torch.FloatTensor(x).type(m.dtype)
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
        # delete hidden state history
        m.lm_hidden = repackage_hidden(m.lm_hidden)
        log_probs = m(x_var)
        training_labels = y_var.view(log_probs.size(0))
        loss = crit(log_probs, training_labels).div(m.batch_size)
        costs += loss.data[0]
        iters += m.num_steps
    cost_norm = (costs/iters)
    ppl = math.exp(min(cost_norm, 100.0))
    return ppl

def test(test_args):

    start = time.time()

    with open(os.path.join(test_args.save_dir, 'config.pkl'), 'rb') as f:
        args = pickle.load(f)

    args.save_dir = test_args.save_dir
    data_loader = TextLoader(args, train=False)
    test_data = data_loader.read_dataset(test_args.test_file)

    print(args.save_dir)
    print("Unit: " + args.unit)
    print("Composition: " + args.composition)

    args.word_vocab_size = data_loader.word_vocab_size
    if args.unit != "word":
        args.subword_vocab_size = data_loader.subword_vocab_size

    # Statistics of words
    print("Word vocab size: " + str(data_loader.word_vocab_size))

    # Statistics of sub units
    if args.unit != "word":
        print("Subword vocab size: " + str(data_loader.subword_vocab_size))
        if args.composition == "bi-lstm":
            if args.unit == "char":
                args.bilstm_num_steps = data_loader.max_word_len
                print("Max word length:", data_loader.max_word_len)
            elif args.unit == "char-ngram":
                args.bilstm_num_steps = data_loader.max_ngram_per_word
                print("Max ngrams per word:", data_loader.max_ngram_per_word)
            elif args.unit == "morpheme" or args.unit == "oracle":
                args.bilstm_num_steps = data_loader.max_morph_per_word
                print("Max morphemes per word", data_loader.max_morph_per_word)

    if args.unit == "word":
        lm_model = WordModel
    elif args.composition == "addition":
        lm_model = AdditiveModel
    elif args.composition == "bi-lstm":
        lm_model = BiLSTMModel
    else:
        sys.exit("Unknown unit or composition.")

    print("Begin testing...")
    mtest = lm_model(args, is_testing=True)
    if args.use_cuda:
        mtest = mtest.cuda()
    # get the last saved model
    model_path, _ = get_last_model_path(args.save_dir)
    saved_model = torch.load(model_path)
    mtest.load_state_dict(saved_model['state_dict'])
    test_perplexity = run_epoch(mtest, test_data, data_loader, eval=True)
    print("Test Perplexity: %.3f" % test_perplexity)
    print("Test time: %.0f\n" % (time.time() - start))
    print("\n")


if __name__ == '__main__':
    main()