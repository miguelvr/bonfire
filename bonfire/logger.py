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

    def __init__(self, metric=None):
        super(BasicLogger, self).__init__()
        self.loss = []
        self.epoch = 0
        self.metric = metric

    def update_on_batch(self, objective):
        self.loss.append(objective)

    def update_on_epoch(self, gold, predictions):
        self.state = None
        self.epoch += 1
        epoch_loss = np.mean(self.loss)
        log = "Epoch: {} | Loss: {}".format(self.epoch, epoch_loss)
        self.loss = []
        if self.metric is not None:
            score = self.metric(gold, predictions)
            log += " | Score: {}".format(score)
        print(log)
