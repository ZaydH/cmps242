#=====================================================================#
# data_parser.py
# functions for text processing for trump speech generator.
#=====================================================================#

import const
import random

def read_input():
    """
    read the input file, get rid of newlines and empty lines, replace with space
    """
    return ' '.join([l.strip() for l in open('trump_speeches.txt').readlines() if l.strip()])


def get_examples(input_string, size):
    """
    from the input, produce examples where the input is a sequence of integers
    representing a string of characters, and the target is the character immediately
    following the input sequence
    """

    sequences = []
    targets = []

    # maps characters in the file to unique integers
    char2int = {c: i for i, c in enumerate(sorted(set(input_string)))}

    if size == -1:

        # get all examples

        # iterate over the file window by window
        i = 0

        while i + const.WINDOW_SIZE + 1 < len(input_string):
            sequences += [[char2int[c] for c in input_string[i: i + const.WINDOW_SIZE]]]
            targets += [char2int[input_string[i + const.WINDOW_SIZE]]]

            i += 1

    else:
        
        # get size many examples
        for z in range(size):

            # get a random starting point
            r = random.choice(range(len(input_string) - const.WINDOW_SIZE - 1))

            # get sequence
            sequences += [[char2int[c] for c in input_string[r: r + const.WINDOW_SIZE]]]

            # get target
            targets += [char2int[input_string[r + const.WINDOW_SIZE]]]

    return [sequences, targets]


# testing
if __name__ == '__main__':
    input_str = read_input()

    print('there are %d unique characters in the input file' % len(set(input_str)))
    print(sorted(set(input_str)))

    X, Y = create_examples(input_str)
