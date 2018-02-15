from bonfire.model import ModelTemplate
from bonfire.logger import LoggerTemplate


class Trainer(object):

    def __init__(self, model, logger):

        assert isinstance(model, ModelTemplate), \
            "model must be a subclass of ModelTemplate, " \
            "received {} instead".format(type(model))

        assert isinstance(logger, LoggerTemplate), \
            "model must be a subclass of LoggerTemplate, " \
            "received {} instead".format(type(logger))

        self.logger = logger
        self.model = model

    def fit(self, train_data, dev_data, epochs=10):
        # Start trainer
        for epoch_n in range(epochs):

            # Train
            for batch in train_data:
                objective = self.model.update(**batch)
                self.logger.update_on_batch(objective)

            # Validation
            predictions = []
            gold = []
            for batch in dev_data:
                predictions.append(self.model.predict(batch['input']))
                gold.append(batch['output'])

            self.logger.update_on_epoch(predictions, gold)

            if self.logger.state == 'save':
                self.model.save()

    def test(self, test_data):
        predictions = []
        for batch in test_data:
            predictions.append(self.model.predict(batch['input']))

        return predictions
