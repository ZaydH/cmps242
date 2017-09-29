import argparse
import logging
import random

import sys
import re

import mock

import numpy as np

import hw1


def setup_logger():
  """
  Setup the logging infrastructure
  """
  logging.basicConfig(format='%(message)s')
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

def parse_args():
  """
  Parses the command line arguments.

  :return: Parsed ArgumentParser namespace.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("training_file",type=str, help="Path to the training set data.")
  parser.add_argument("test_file", type=str, help="Path to the test set data.")
  parser.add_argument("results_file", type=str, help="Path to write the results output.")

  parser.add_argument("min_degree",type=int, help="Minimum polynomial degree to test (inclusive).")
  parser.add_argument("max_degree", type=int, help="Maximum polynomial degree to test (inclusive).")

  parser.add_argument("--leave_one_out", action="store_true", help="Perform leave-one-out cross validation.")
  parser.add_argument("-k", type=int, help="Number of folds for cross validation")

  args = parser.parse_args()

  # Verify the input argument validity
  if args.min_degree < 0 or args.max_degree < 0:
    report_input_arg_error("Polynomial degree must be greater than or equal to 0.")
  if args.max_degree < args.min_degree:
    report_input_arg_error("Maximum polynomial degree must be greater than the minimum")

  # Check the optional input arguments
  numb_opt_args = 0
  if args.k:
    numb_opt_args += 1
  if args.leave_one_out:
    numb_opt_args += 1
  if numb_opt_args != 1:
    report_input_arg_error("Exactly one optional input argument must be passed.")

  return args


def report_input_arg_error(msg):
  """
  Reports an input error then exits the program.

  :param msg: Error message to print for the input_arguments
  :type msg: str
  """
  logging.error(msg)
  sys.exit(-1)


def parse_data_file(file_path):
  """
  Parses a test/training data file and returns the data in input/output value pairs.

  :param file_path: Location of the text file
  :type file_path: str

  :return: List of pairs of the input value and expected values
  :rtpe: List[Tuple[int]]
  """
  # Use with to eliminate the need to close
  with open(file_path, 'r') as f:
    file_contents = [tuple(re.split("\s+", line)) for line in f.read().splitlines()]
    shuffled_data = [[float(str_tuple[0]), float(str_tuple[1])] for str_tuple in file_contents]
    random.shuffle(shuffled_data)  # only done in place.  Randomizes the order
    file_data = np.matrix(shuffled_data, dtype=np.float32)
  return file_data

if __name__ == "__main__":
  """
  Run the solver
  """
  setup_logger()
  args = parse_args()

  # Parse the input data files
  params = mock.Mock()
  params.training_data = parse_data_file(args.training_file)
  params.test_data = parse_data_file(args.test_file)

  params.results_file = args.results_file

  # Store the degrees to test
  params.min_degree = args.min_degree
  params.max_degree = args.max_degree

  # Determine the number of folds
  if args.leave_one_out:
    params.k = len(params.training_data)
  elif args.k:
    params.k = args.k

  hw1.run(params)
