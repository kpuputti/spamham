"""\
spamham.py - spam/ham classifier

== Usage: ==
python spamham.py [method] [args]

where [method] is the task to be done (see below), and [args] is the
arguments for the task.

== Methods: ==
1. Train the classifier with a data file and output its success:
python spamham.py train [classifier] [train_file]

2. Train and classify given data into an output file:
python spamham.py classify [classifier] [train_file] [data_file] [output_file]

3. Validate generated output file against a labeled data file:
python spamham.py validate [output_file] [labled_file]"""
import classifiers
import sys


class UnknownClassifierError(Exception):
    pass


def usage():
    """Print the usage help from module doc."""
    print __doc__


def train(classifier_name, data_file):
    if not hasattr(classifiers, classifier_name):
        raise UnknownClassifierError('Unknown classifier: %s' % classifier_name)
    classifier = getattr(classifiers, classifier_name)(data_file)
    print 'Training classifier:', classifier.name
    classifier.train()


def classify(classifier_name, train_file, data_file, output_file):
    if not hasattr(classifiers, classifier_name):
        raise UnknownClassifierError('Unknown classifier: %s' % classifier_name)
    classifier = getattr(classifiers, classifier_name)(data_file)
    print 'Classifying data with classifier:', classifier.name


def validate(output_file, labeled_file):
    print 'Validating classified output file:', output_file


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

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
