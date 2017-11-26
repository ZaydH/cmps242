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
    with open(Config.Train.training_file, "r") as f:
        input_text = f.read()
    return ' '.join(input_text.splitlines())


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

    assert(len(sequences) == len(targets))
    # Define how to randomly split the input data into train and test
    shuffled_list = [i for i in range(0, len(sequences))]
    random.shuffle(shuffled_list)
    split_point = int(Config.training_split_ratio * len(sequences))

    Config.Train.train_x = [sequences[idx] for idx in shuffled_list[:split_point]]
    Config.Train.train_t = [targets[idx] for idx in shuffled_list[:split_point]]

    Config.Train.verify_x = [sequences[idx] for idx in shuffled_list[split_point:]]
    Config.Train.verify_t = [targets[idx] for idx in shuffled_list[split_point:]]


def build_training_and_verification_sets(dataset_size=-1):
    if Config.Train.restart:
        input_str = read_input()
        create_examples(input_str, dataset_size)
        # Character to integer map required during text generation
        Config.export_character_to_integer_map()
        # Export the training and verification data in case
        # the previous setup will be trained on aga
        Config.export_train_and_verification_data()
    else:
        Config.import_character_to_integer_map()
        Config.import_train_and_verification_data()

    # Print basic statistics on the training set
    logging.info("Vocabulary Size: \t\t%d" % Config.vocab_size())
    logging.info("Training Set Size: \t\t%d" % len(Config.Train.train_t))
    logging.info("Verification Set Size: \t%d" % len(Config.Train.verify_t))


# testing
if __name__ == '__main__':
    build_training_and_verification_sets()
