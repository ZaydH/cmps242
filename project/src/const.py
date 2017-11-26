import logging
import argparse
import __main__
import os
import pickle

from decision_engine import DecisionFunction


class Config(object):

  """
  Directory to export the trained model.
  """
  model_dir = './model/'
  """
  Character to integer look up.
  """
  char2int = None
  """
  Character dictionary exported to a pickle file so that it
  does not need to be reconstructed during text generation.
  """
  char2int_pk_file = "char2int.pk"

  WINDOW_SIZE = 50
  EMBEDDING_SIZE = 30
  RNN_HIDDEN_SIZE = 64

  """
  Stores whether training is being executed.
  """
  _is_train = False
  """
  Name of the main file.
  """
  _main = ""

  class Train(object):
    """
    Stores all configuration settings and objects related to the training of
    the neural network.
    """

    """
    File containing the text training set.
    """
    training_file = "." + os.sep + "trump_speeches.txt"
    """
    Object containing the input training set.
    """
    inputs = None
    """
    Pickle file to export the input training set
    """
    inputs_pk_file = "inputs.pk"

    targets = None
    targets_pk_file = "targets.pk"
    epochs = 100
    batch_size = 100

    """
    If true, restart the training from scratch. 
    """
    restart = True

  class FF(object):
    """
    Configuration settings for the feed-forward network.
    """
    depth = 1
    hidden_width = 128

  class DecisionEngine(object):
    """
    Configuration settings for the decision engine.
    """
    function = DecisionFunction.ArgMax

  @staticmethod
  def main():
    """
    Main Python file

    :return: Name of the main file.
    :rtype: str
    """
    if not Config._main:
      sep_loc = __main__.__file__.rfind(os.sep)
      Config._main = __main__.__file__[sep_loc + 1:]
    return Config._main

  @staticmethod
  def vocab_size():
    """
    Vocabulary size accessor.

    :return: Number of characters in the input and output vocabulary.
    """
    return len(Config.char2int)

  @staticmethod
  def parse_args():
    """
    Input Argument Parser

    Parses the command line input arguments.
    """
    # Select the arguments based on what program is running
    if Config.main() == "train.py":
      Config._is_train = True
      Config._train_args()
    elif Config.main() == "trump.py":
      Config._is_train = False
      Config._trump_args()
    else:
      raise ValueError("Unknown main file.")

  @staticmethod
  def _train_args():
    parser = argparse.ArgumentParser("Character-Level RNN Trainer")
    parser.add_argument("--train", type=str, required=False,
                        default=Config.Train.training_file,
                        help="Path to the training set file.")
    parser.add_argument("--use_existing", action="store_true",
                        help="Continue training the existing model")
    parser.add_argument("--model", type=str, required=False,
                        default=Config.model_dir,
                        help="Directory to which to export the trained network")
    parser.add_argument("--seqlen", type=int, required=False,
                        default=Config.WINDOW_SIZE,
                        help="RNN sequence length")
    parser.add_argument("--epochs", type=int, required=False,
                        default=Config.Train.epochs,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, required=False,
                        default=Config.Train.batch_size,
                        help="Batch size")
    args = parser.parse_args()

    Config.WINDOW_SIZE = args.seqlen

    Config.Train.training_file = args.train
    Config.Train.restart = not args.use_existing
    Config.Train.epochs = args.epochs
    Config.Train.batch_size = args.batch

  @staticmethod
  def _trump_args():
    parser = argparse.ArgumentParser("Character-Level Trump Text Generator")
    parser.add_argument("--model", type=str, required=True,
                        desc="Directory containing the trained model")
    parser.add_argument("--seed", type=str, required=True,
                        desc="Text with which to seed the generator")
    parser.add_argument("--decision", type=int, required=False,
                        default=DecisionFunction.ArgMax,
                        desc="Function of the decision engine.  Set to \"0\" to always select "
                             + "the character with maximum probability. Set to \"1\" to make a "
                             + "weighted random selection for the first character after a space "
                             + "and then use argmax")

  @staticmethod
  def import_training_data():
    logging.info("Importing the training dataset and the character to integer map.")
    Config.Train.inputs = _pickle_import(Config.Train.inputs_pk_file)
    Config.Train.targets = _pickle_import(Config.Train.targets_pk_file)
    Config.char2int = _pickle_import(Config.char2int_pk_file)
    logging.info("COMPLETED: Importing the training dataset.")

  @staticmethod
  def import_model():
    logging.info("Importing the trained model...")
    # ToDo Implement importing the TensorFlow model
    logging.info("COMPLETED: Importing the trained model")

  @staticmethod
  def export_model():
    logging.info("Checkpoint: Exporting the trained model...")
    # ToDo Implement exporting the TensorFlow model
    logging.info("COMPLETED Checkpoint: Exporting the trained model")

  @staticmethod
  def is_train():
    if not Config._main:
      Config.main()
    return Config._is_train


def _pickle_export(obj, file_name):
  """
  Pickle Exporter

  Pickles the specified object and writes it to the specified file name.

  :param obj: Object to be pickled.
  :type obj: Object

  :param file_name: File to write the specified object to.
  :type file_name: str
  """
  pickle.dump(obj, open(file_name, "wb"))


def _pickle_import(file_name):
  """
  Pickle Importer

  Helper function for importing pickled objects.

  :param file_name: Name and path to the pickle file.
  :type file_name: str

  :return: The pickled object
  :rtype: Object
  """
  return pickle.load(open(file_name, "rb"))
