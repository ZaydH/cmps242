"""
    data_parser.py

    Functions for text processing for trump speech generator.
"""
import logging
import random
from const import Config


def read_input():
    """
    Reads the input file, get rid of newlines and empty lines, replace with space
    """
    return ' '.join([l.strip() for l in open(Config.Train.training_file).readlines() if l.strip()])


def create_examples(input_string, size=-1):
    """
    from the input, produce examples where the input is a sequence of integers
    representing a string of characters, and the target is the character immediately
    following the input sequence
    """

    sequences = []
    targets = []
    Config.char2int = {c: i for i, c in enumerate(sorted(set(input_string)))}

    # Get all examples
    if size == -1:
        # iterate over the file window by window
        i = 0
        while i + Config.WINDOW_SIZE + 1 < len(input_string):
            sequences += [[Config.char2int[c] for c in input_string[i: i + Config.WINDOW_SIZE]]]
            targets += [Config.char2int[input_string[i + Config.WINDOW_SIZE]]]

            i += 1

    else:
        # get size many examples
        for z in range(size):

            # get a random starting point
            r = random.choice(range(len(input_string) - Config.WINDOW_SIZE - 1))

            # get sequence
            sequences += [[Config.char2int[c] for c in input_string[r: r + Config.WINDOW_SIZE]]]

            # get target
            targets += [Config.char2int[input_string[r + Config.WINDOW_SIZE]]]

    Config.Train.inputs = sequences
    Config.Train.targets = targets


def build_training_set():
    if Config.Train.restart:
        input_str = read_input()
        create_examples(input_str)
    else:
        Config.import_training_data()

    logging.info("Training Set Size: \t%d" % len(Config.Train.targets))
    logging.info("Vocabulary Size: \t\t%d" % len(set(input_str)))


# testing
if __name__ == '__main__':
    build_training_set()
