import math
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm

class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm, lr_decay=1, patience=None):
        self.last_ppl = 10000000.0
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.patience = patience
        self.start_decay = False
        self.minrun = self.patience+1

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    # I may update this function
    # the number of iterations allowed before decaying the learning rate if there is no improvement on dev set
    def updateLearningRate(self, ppl, epoch):
        self.start_decay = False
        diff = self.last_ppl-ppl
        if diff < 0.1:
            self.start_decay = True
        # If no improvement and it has run for minimum required epoch
        # decrease patience
        if self.patience is not None and \
                epoch >= self.minrun and \
                         self.start_decay:
            self.patience-=1
        # If have no patience for this thing
        if self.patience==0 and self.start_decay:
            self.lr = self.lr * self.lr_decay
            # reset patience for the new learning rate
            self.patience = self.minrun-1
            print("Decaying learning rate to %g" % self.lr)
        # register new values
        self.last_ppl = ppl
        self.optimizer.param_groups[0]['lr'] = self.lr