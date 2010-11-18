"""\
spamham.py - spam/ham classifier

Usage:
python spamham.py [classifier] [args]

where [classifier] is the name of the classifier and [args] is a list
of arguments for the classifier."""
import sys


def usage():
    """Print the usage help from module doc."""
    print __doc__


def main(args):
    """Dispatch the functionality based on the argument list."""
    if len(args) == 0:
        sys.stderr.write('Error: No method specified\n')
        usage()
        sys.exit(2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
