"""\
spamham.py - spam/ham classifier

== Usage: ==
python spamham.py [method] [args]

where [method] is the task to be done (see below), and [args] is the
list of arguments for the task.

== Methods: ==
1. Train the classifier with a data file and output its success:
python spamham.py train [classifier] [train_file]

2. Train and classify given data into an output file:
python spamham.py classify [classifier] [train_file] [data_file] [output_file]

3. Validate generated output file against a labeled data file:
python spamham.py validate [output_file] [labled_file]"""
import classifiers
import itertools
import logging
import sys


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger('spamham')


class UnknownClassifierError(Exception):
    pass


def usage():
    """Print the usage help from module doc."""
    print __doc__


def read_data(file_name):
    """Read the space separated data from the given file."""
    logger.info('Reading data from file: ' + file_name)
    data = []
    with open(file_name) as f:
        for line in f:
            datum = line.strip().split()
            if datum[1] == 'nan':
                # If the label is unknown, it is marked as 'nan'.
                datum[1] = None
                data.append([int(d) for d in datum if d is not None])
            else:
                data.append([int(d) for d in datum])

    return data


def train(classifier_name, data_file):
    if not classifier_name.endswith('Classifier') \
           or not hasattr(classifiers, classifier_name):
        raise UnknownClassifierError('Unknown classifier: %s' % classifier_name)
    logger.info('Training classifier: ' + classifier_name)
    data = read_data(data_file)
    classifier = getattr(classifiers, classifier_name)(data)
    classifier.train()

    correct = 0
    incorrect = 0
    for datum in classifier.validationset:
        is_spam = datum[1]
        if classifier.classify(datum) == is_spam:
            correct += 1
        else:
            incorrect += 1

    correct_percentage = 100 * float(correct) / (correct + incorrect)

    # Print statistics of the classification.
    print '== Training output for classifier:', classifier.name, ' =='
    print 'total data length:', len(data)
    print 'train data length:', len(classifier.trainset)
    print 'validation data length:', len(classifier.validationset)
    print 'test data length:', len(classifier.testset)
    print 'correct:', correct
    print 'incorrect:', incorrect
    print 'correctness percentage: %.2f%%' % correct_percentage


def classify(classifier_name, train_file, data_file, output_file):
    if not classifier_name.endswith('Classifier') \
           or not hasattr(classifiers, classifier_name):
        raise UnknownClassifierError('Unknown classifier: %s' % classifier_name)
    logger.info('Classifying data with classifier:' + classifier_name)
    traindata = read_data(train_file)
    classifier = getattr(classifiers, classifier_name)(traindata, classify=True)
    classifier.train()
    data = read_data(data_file)
    logger.info('Writing classification results to file: ' + output_file)
    with open(output_file, 'w') as out:
        for datum in data:
            is_spam = classifier.classify(datum)
            out.write('%d %d\n' % (datum[0], is_spam))


def validate(output_file, labeled_file):
    logger.info('Validating classified output file:' + output_file)
    output = read_data(output_file)
    labeled = read_data(labeled_file)
    assert len(output) == len(labeled)
    datums = len(output)
    matches = 0
    false_positives = 0
    for d1, d2 in itertools.izip(output, labeled):
        assert d1[0] == d2[0], 'Datum ids must match!'
        outlabel = d1[1]
        labeled_label = d2[1]
        if outlabel == labeled_label:
            matches += 1
        elif outlabel and not labeled_label:
            false_positives += 1

    print '== Validation output: =='
    print 'datums:', datums
    print 'accurate matches:', matches
    print 'accuracy percentage: %.2f%%' % (100 * float(matches) / datums)
    print 'false positives:', false_positives
    print 'false positive percentage: %.2f%%' \
          % (100 * float(false_positives) / datums)


def main(args):
    """Dispatch the functionality based on the argument list."""
    if len(args) < 3:
        sys.stderr.write('Error: Not enough arguments\n')
        usage()
        return 2

    method = args.pop(0)

    if method == 'train':
        train(*args)
    elif method == 'classify':
        classify(*args)
    elif method == 'validate':
        validate(*args)
    else:
        sys.stderr.write('Error: Unknown method: %s\n' % method)
        usage()
        return 1

    print 'Done.'
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
