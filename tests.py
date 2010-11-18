import classifiers
import spamham
import unittest
import sys
from StringIO import StringIO


def get_data(length):
    """Generate a data set with the given amount of datums."""
    return [(i, str(i)) for i in xrange(1, length + 1)]


class SpamhamTest(unittest.TestCase):

    def test_main__with_not_enough_arguments(self):
        """Test the return value with too few arguments."""
        self.assertEquals(spamham.main([]), 2)
        self.assertEquals(spamham.main(['arg1']), 2)
        self.assertEquals(spamham.main(['arg1', 'arg2']), 2)

    def test_main__train_with_invalid_classifier(self):
        """Test the train method with an invalid classifier."""
        func = lambda: spamham.main(['train', 'InvalidClassifier', 'trainfile'])
        self.assertRaises(spamham.UnknownClassifierError, func)

    def test_main__classify_with_invalid_classifier(self):
        """Test the classify method with an invalid classifier."""
        func = lambda: spamham.main(['classify', 'InvalidClassifier',
                                     'train', 'data', 'output'])
        self.assertRaises(spamham.UnknownClassifierError, func)

    def test_main__with_unknown_method(self):
        """Test the return value with an unknown method."""
        self.assertEquals(spamham.main(['unknown', 'arg1', 'arg2']), 1)


class ClassifierTest(unittest.TestCase):

    def test_split_data__len(self):
        """Test that the split returns the correct amount of items."""
        data = get_data(5)
        classifier = classifiers.Classifier([])
        trainset, testset = classifier.split_data(data)
        self.assertEquals(len(data), len(trainset) + len(testset))

    def test_split_data__items(self):
        """Test that the split returns all and noting but the original
        items."""
        data = get_data(20)
        classifier = classifiers.Classifier([])
        trainset, testset = classifier.split_data(data)
        ids = set(d[0] for d in data)
        self.assertEquals(ids, trainset | testset)


if __name__ == '__main__':
    # Catch stdout and stderr.
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    unittest.main()
