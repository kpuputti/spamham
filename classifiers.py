import random


class Classifier(object):
    """Base class for all classifiers."""

    name = 'base classifier'

    def __init__(self, data):
        self.data = data
        self.trainset, self.validationset, self.testset = self.split_data(data)

    def split_data(self, data):
        """Split the data set into training, validation, and test
        sets."""
        trainset = []
        validationset = []
        testset = []
        for datum in data:
            if datum[1] is None:
                testset.append(datum)
            elif random.choice((True, False)):
                trainset.append(datum)
            else:
                validationset.append(datum)
        return trainset, validationset, testset

    def train(self):
        raise NotImplementedError('Implement this method in a derived class.')

    def classify(self, datum):
        """Return a Boolean value indicating whether the given datum
        is spam or ham."""
        raise NotImplementedError('Implement this method in a derived class.')


class RandomClassifier(Classifier):
    """Classify data randomly."""

    name = 'random classifier'

    def train(self):
        pass

    def classify(self, datum):
        return random.choice((True, False))


class SVMClassifier(Classifier):
    """Classify the data using Support Vector Machines."""

    name = 'SVM classifier'

    def train(self):
        pass

    def classify(self):
        pass
