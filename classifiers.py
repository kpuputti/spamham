import random


class Classifier(object):
    """Base class for all classifiers."""

    def __init__(self, trainset):
        self.trainset = trainset

    def train(self):
        raise NotImplementedError('Implement this method in a derived class.')

    def classify(self, datum):
        raise NotImplementedError('Implement this method in a derived class.')


class RandomClassifier(Classifier):
    """Classify data randomly."""

    def train(self):
        pass

    def classify(self, datum):
        return random.choice((True, False))


class SVMClassifier(Classifier):
    """Classify the data using Support Vector Machines."""

    def train(self):
        pass

    def classify(self):
        pass
