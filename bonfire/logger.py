from abc import ABCMeta, abstractmethod


class LoggerTemplate(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def update_on_batch(self, *args):
        raise NotImplementedError

    @abstractmethod
    def update_on_epoch(self, *args):
        raise NotImplementedError


class BasicLogger(LoggerTemplate):

    def __init__(self):
        pass

    def update_on_batch(self, *args):
        pass

    def update_on_epoch(self, *args):
        pass
