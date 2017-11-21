# =====================================================================#
# parser.py
# functions for text processing for trump speech generator.
# =====================================================================#

import const

def read_input():
    '''
    read the input file, get rid of newlines and empty lines, replace with space
    '''
    return ' '.join([l.strip() for l in open('trump_speeches.txt').readlines() if l.strip()])

def create_examples(input_str):
    '''
    from the input, produce examples where the input is a sequence of integers
    representing a string of characters, and the target is the character immediately
    following the input sequence
    '''

    sequences = []; targets = [];

    # maps characters in the file to unique integers
    char2int = {c: i for i,c in enumerate(sorted(set(input_str)))}

    # iterate over the file window by window
    i = 0
    
    while i + const.WINDOW_SIZE + 1 < len(input_str):

        sequences += [[char2int[c] for c in input_str[i: i + const.WINDOW_SIZE]]]
        targets += [char2int[input_str[i + const.WINDOW_SIZE]]]

        i += 1

    return sequences, targets

# testing
if __name__ == '__main__':
    
    input_str = read_input()

    print('there are %d unique characters in the input file' % len(set(input_str)))
    print(sorted(set(input_str)))

    sequences, targets = create_examples(input_str)