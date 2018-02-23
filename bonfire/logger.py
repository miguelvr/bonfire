import logging
import numpy as np
from abc import ABCMeta, abstractmethod


class LoggerTemplate(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.state = None

    @abstractmethod
    def update_on_batch(self, *args):
        raise NotImplementedError

    @abstractmethod
    def update_on_epoch(self, *args):
        raise NotImplementedError


class BasicLogger(LoggerTemplate):

    def __init__(self, metric=None, score_optimization='min'):
        super(BasicLogger, self).__init__()
        self.loss = []
        self.epoch = 0
        self.metric = metric
        self.score_optimization = score_optimization
        if self.score_optimization == 'min':
            self.best_score = np.inf
        elif self.score_optimization == 'max':
            self.best_score = -np.inf
        else:
            raise ValueError("score_optimization must be either 'min' or 'max'")

    def update_on_batch(self, objective):
        self.loss.append(objective)

    def update_on_epoch(self, gold, predictions):
        self.state = None
        self.epoch += 1
        epoch_loss = np.mean(self.loss)
        log = "Epoch: {} | Loss: {:.3f}".format(self.epoch, epoch_loss)
        self.loss = []
        if self.metric is not None:
            score = self.metric(gold, predictions)
            log += " | Score: {:.3f}".format(score)

            if self.score_optimization == 'min':
                improved = score < self.best_score
            else:
                improved = score > self.best_score

            if improved:
                self.best_score = score
                self.state = 'save'
        else:
            self.state = 'save'
        print(log)
