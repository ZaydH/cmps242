#=====================================================================#
# parser.py
# functions for text processing for trump speech generator.
#=====================================================================#

def read_input():
	'''
	read the input file, get rid of newlines and empty lines, replace with space
	'''
	return ' '.join([l.strip() for l in open('trump_speeches.txt').readlines() if l.strip()])

# testing
if __name__ == '__main__':

	# get the input string
	input_str = read_input()