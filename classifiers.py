import PyML
import collections
import logging
import random


logger = logging.getLogger('classifiers')


class Classifier(object):
    """Base class for all classifiers."""

    name = 'base classifier'

    def __init__(self, data, classify=False):
        logging.debug('Creating classifier: ' + self.name)
        self.classify_mode = classify
        self.trainset, self.validationset, self.testset = self.split_data(data)

    def split_data(self, data):
        """Split the data set into training, validation, and test
        sets."""
        logger.debug('Splitting the training data.')
        trainset = []
        validationset = []
        testset = []
        for datum in data:
            if datum[1] is None:
                testset.append(datum)
            elif self.classify_mode or random.choice((True, False)):
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


class AllSpamClassifier(Classifier):
    """Classify all data as spam."""

    name = 'all spam classifier'

    def train(self):
        pass

    def classify(self, datum):
        return True


class AllHamClassifier(Classifier):
    """Classify all data as ham."""

    name = 'all ham classifier'

    def train(self):
        pass

    def classify(self, datum):
        return False


class DummyClassifier(Classifier):
    """Uses some dummy logic for classification."""

    name = 'dummy classifier'

    # Frequency in all spam messages needed for a field to be a spam
    # field.
    SPAM_FIELD_FREQ = 0.5
    # Frequency of spam fields needed within all spam fields for a
    # datum to be a spam datum.
    SPAM_FIELDS_FREQ = 0.5

    def __init__(self, data, classify=False):
        self.spam_colums = None
        super(DummyClassifier, self).__init__(data, classify)

    def train(self):
        logger.debug('Training classifier: ' + self.name)
        num_spams = 0
        spam_columns = collections.defaultdict(int)
        for datum in self.trainset:
            # Skip ham datums.
            if datum[1] == 0:
                continue
            num_spams += 1
            for i in xrange(2, len(datum)):
                if datum[i]:
                    spam_columns[i] += 1
        spam_freq_needed = self.SPAM_FIELD_FREQ * num_spams
        spams = set()
        # Add all columns with a big enough count to spam column.
        for column, count in spam_columns.iteritems():
            if count >= spam_freq_needed:
                spams.add(column)
        self.spam_columns = spams


    def classify(self, datum):
        assert self.spam_columns is not None, 'Train the classifier first!'
        spam_columns_needed = self.SPAM_FIELDS_FREQ * len(self.spam_columns)
        num_spam_columns = 0
        for column in self.spam_columns:
            if datum[column]:
                num_spam_columns += 1
        return num_spam_columns >= spam_columns_needed


class SVMClassifier(Classifier):
    """Classify the data using Support Vector Machines."""

    name = 'SVM classifier'

    def __init__(self, data, classify=False, C=None):
        super(SVMClassifier, self).__init__(data, classify)
        self.convert_data()
        if C is None:
            self.svm = PyML.SVM()
        else:
            self.svm = PyML.SVM(C=C)

    def convert_data(self):
        """Convert the data sets into a form expected by PyML."""
        logger.info('Converting classifier data into PyML format.')
        data = []
        for datum in self.trainset:
            data.append([int(d) for d in datum[2:]])
        svm_trainset = PyML.VectorDataSet(data)
        labels = PyML.Labels([str(d[1]) for d in self.trainset])
        svm_trainset.attachLabels(labels)
        self.svm_trainset = svm_trainset

    def train(self):
        self.svm.train(self.svm_trainset)

    def classify(self, datum):
        cols = [int(d) for d in datum[2:]]
        test_datum = PyML.VectorDataSet([cols])
        is_spam = self.svm.classify(test_datum, 0)
        return bool(is_spam[0])

class DerivedSVMClassifier(SVMClassifier):

    def __init__(self, data, classify=False):
        C = 0.1
        super(DerivedSVMClassifier, self).__init__(data, classify, C)
