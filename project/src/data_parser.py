"""
    data_parser.py

    Functions for text processing for trump speech generator.
"""
import logging
import random
import numpy as np
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
    depths = []
    Config.char2int = {c: i for i, c in enumerate(sorted(set(input_string)))}

    # ToDo Discuss with Ben how we want to train on text shorter than the window size?

    # Get all examples
    if size == -1:
        # iterate over the file window by window
        i = 0
        while i + Config.sequence_length + 1 < len(input_string):
            sequences += [[Config.char2int[c] for c in input_string[i: i + Config.sequence_length]]]
            targets += [Config.char2int[input_string[i + Config.sequence_length]]]
            i += 1

    else:
        # get size many examples
        for z in range(size):

            # get a random starting point
            r = random.choice(range(len(input_string) - Config.sequence_length - 1))

            sequences.append([Config.char2int[c] for c in input_string[r: r + Config.sequence_length]])
            depths.append(Config.sequence_length)
            targets.append(Config.char2int[input_string[r + Config.sequence_length]])

    assert(len(sequences) == len(targets))
    # Define how to randomly split the input data into train and test
    shuffled_list = list(range(len(sequences)))
    random.shuffle(shuffled_list)
    split_point = int(Config.training_split_ratio * len(sequences))

    Config.Train.x = [sequences[idx] for idx in shuffled_list[:split_point]]
    # Config.Train.t = [targets[idx] for idx in shuffled_list[:split_point]]
    Config.Train.depth = [depths[idx] for idx in shuffled_list[:split_point]]
    # Config.Train.x = list(map(lambda idx: _build_input_sequence(sequences[idx]),
    #                           shuffled_list[:split_point]))
    Config.Train.t = list(map(lambda idx: _build_target_vector(targets[idx]),
                              shuffled_list[:split_point]))

    Config.Verify.x = [sequences[idx] for idx in shuffled_list[split_point:]]
    # Config.Verify.t = [targets[idx] for idx in shuffled_list[:split_point]]
    Config.Verify.depth = [depths[idx] for idx in shuffled_list[split_point:]]
    # Config.Verify.x = list(map(lambda idx: _build_input_sequence(sequences[idx]),
    #                            shuffled_list[split_point:]))
    Config.Verify.t = list(map(lambda idx: _build_target_vector(targets[idx]),
                               shuffled_list[split_point:]))


def _build_input_sequence(int_sequence):
    """
    One-Hot Sequence Builder

    Converts a list of integers into a sequence of integers.

    :param int_sequence: List of the character indices
    :type int_sequence: List[int]

    :return: Input sequence converted into a matrix of one hot rows
    :rtype: np.ndarray
    """
    assert(0 < len(int_sequence) <= Config.sequence_length)
    one_hots = []
    while len(one_hots) < Config.sequence_length:
        idx = len(one_hots)
        char_id = 0  # This is used to pad the list as needed
        if idx < len(int_sequence):
            char_id = int_sequence[idx]
        vec = np.zeros([Config.vocab_size()])
        vec[char_id] = 1
        one_hots.append(vec)
    seq = np.vstack(one_hots)
    return seq


def _build_target_vector(idx):
    """
    Creates a one hot vector for the target with "1" in the correct character
    location and zero everywhere else.

    :param idx: Integer corresponding to the expected character
    :type idx: int

    :return: One hot vector for the target character
    :rtype: np.array
    """
    assert(0 <= idx < Config.vocab_size())
    one_hot = np.zeros([Config.vocab_size()])
    one_hot[idx] = 1
    return one_hot


def build_training_and_verification_sets(dataset_size=-1):
    """
    Training and Verification Set Builder

    Builds the training and verification datasets.  Depending on the
    configuration, this may be from the source files or from pickled
    files.

    :param dataset_size: Number of total elements in the training and verification sets.
    :type dataset_size: int
    """
    if not Config.Train.restore:
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
    logging.info("Training Set Size: \t\t%d" % len(Config.Train.t))
    logging.info("Verification Set Size: \t%d" % len(Config.Verify.t))


# testing
if __name__ == '__main__':
    build_training_and_verification_sets()
