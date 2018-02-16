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

    def __init__(self):
        super(BasicLogger, self).__init__()

    def update_on_batch(self, *args):
        pass

    def update_on_epoch(self, *args):
        pass
