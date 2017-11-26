import logging
import argparse
import __main__
import os
import pickle
import tensorflow as tf
import math

from decision_engine import DecisionFunction


class Config(object):
  """
  Master configuration class containing all settings related to
  the network, training, etc.
  """

  """
  Directory to export the trained model.
  """
  model_dir = '.' + os.sep + 'model' + os.sep
  """
  Name assigned to the TensorFlow model
  """
  model_name = "trump"
  """
  Character to integer look up.
  """
  char2int = None
  """
  Character dictionary exported to a pickle file so that it
  does not need to be reconstructed during text generation.
  """
  char2int_pk_file = "char2int.pk"

  sequence_length = 50
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
  """
  Split between training and verification
  sets.
  """
  training_split_ratio = 0.8

  class Verify(object):
    x = None
    t = None
    """
    For each training object, it is the number of vectors before
    the output is expected
    """
    depth = None
    """
    Pickle file to store the verify_x and verify_t objects.
    """
    pk_file = "verify.pk"

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
    Input training data
    """
    x = None
    """
    Training Labels
    """
    t = None
    """
    For each training object, it is the number of vectors before
    the output is expected
    """
    depth = None
    """
    Pickle file to export the input training set
    """
    pk_file = "train.pk"

    num_epochs = 100
    """
    Number of elements per batch.
    """
    batch_size = 100
    """
    If true, restore the previous settings
    """
    restore = True
    """
    Number of epochs between model checkpoint.
    """
    checkpoint_frequency = 10
    learning_rate = 1.0
    _num_batch = -1

    @staticmethod
    def size():
      """
      Number of elements in the training set

      :return: Size of the training set
      :rtype: int
      """
      return len(Config.Train.t)

    @staticmethod
    def num_batch():
      if Config.Train._num_batch <= 0:
        Config.Train._num_batch = int(math.ceil(Config.Train.size() /
                                                Config.Train.batch_size))
      return Config.Train._num_batch

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
      Config._main = os.path.basename(__main__.__file__)
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
    Config.setup_logger()

  @staticmethod
  def _train_args():
    """
    Training Command Line Argument Parser

    Parsers the command line arguments when performing training.
    """
    parser = argparse.ArgumentParser("Character-Level RNN Trainer")
    parser.add_argument("--train", type=str, required=False,
                        default=Config.Train.training_file,
                        help="Path to the training set file.")
    parser.add_argument("--restore", action="store_true",
                        help="Continue training the existing model")
    parser.add_argument("--model", type=str, required=False,
                        default=Config.model_dir,
                        help="Directory to which to export the trained network")
    parser.add_argument("--seqlen", type=int, required=False,
                        default=Config.sequence_length,
                        help="RNN sequence length")
    parser.add_argument("--epochs", type=int, required=False,
                        default=Config.Train.num_epochs,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, required=False,
                        default=Config.Train.batch_size,
                        help="Batch size")
    args = parser.parse_args()

    Config.sequence_length = args.seqlen

    Config.model_dir = args.model
    if Config.model_dir[-1] != os.sep:
      Config.model_dir += os.sep

    Config.Train.training_file = args.train
    Config.Train.restore = args.restore
    Config.Train.num_epochs = args.epochs
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
  def import_train_and_verification_data():
    logging.info("Importing the training and verification datasets.")
    Config.Train.x, Config.Train.t, Config.Train.depth \
        = _pickle_import(Config.model_dir + Config.Train.pk_file)
    Config.Verify.x, Config.Verify.t, Config.Verify.depth \
        = _pickle_import(Config.model_dir + Config.Verify.pk_file)
    logging.info("COMPLETED: Importing the training and verificationdataset.")

  @staticmethod
  def export_train_and_verification_data():
    logging.info("Importing the training dataset and the character to integer map.")
    _pickle_export([Config.Train.x, Config.Train.t, Config.Train.depth],
                   Config.model_dir + Config.Train.pk_file)
    _pickle_export([Config.Verify.x, Config.Verify.t, Config.Verify.depth],
                   Config.model_dir + Config.Verify.pk_file)
    logging.info("COMPLETED: Importing the training dataset.")

  @staticmethod
  def export_character_to_integer_map():
    logging.info("Exporting the character to integer map...")
    _pickle_export(Config.char2int, Config.model_dir + Config.char2int_pk_file)
    logging.info("COMPLETED: Exporting the character to integer map")

  @staticmethod
  def import_character_to_integer_map():
    logging.info("Importing the character to integer map...")
    Config.char2int = _pickle_import(Config.model_dir + Config.char2int_pk_file)
    logging.info("COMPLETED: Importing the character to integer map")

  @staticmethod
  def import_model(sess):
    """
    Imports the weights of the training network.  This can be used
    to continue training or when generating text.

    :param sess: TensorFlow session to which to restore
    :type sess: tf.Session
    """
    logging.info("Importing the trained model...")
    model_file = (Config.model_dir + Config.model_name
                  + "-" + str(Config.Train.checkpoint_frequency) + ".meta")
    new_saver = tf.train.import_meta_graph(model_file)
    new_saver.restore(sess, tf.train.latest_checkpoint(Config.model_dir))
    logging.info("COMPLETED: Importing the trained model")

  @staticmethod
  def export_model(sess, epoch):
    """
    Exports the network weights.
    """
    logging.info("Checkpoint: Exporting the trained model...")
    saver = tf.train.Saver(max_to_keep=20)
    # Only write the meta for the first checkpoint
    write_meta = (not Config.Train.restore) and (epoch == Config.Train.checkpoint_frequency)
    saver.save(sess, Config.model_name, global_step=epoch,
               write_meta_graph=write_meta)
    # ToDo Implement exporting the TensorFlow model
    logging.info("COMPLETED Checkpoint: Exporting the trained model")

  @staticmethod
  def is_train():
    """
    Gets whether the current run is training.

    :return: true if training is being performed.
    :rtype: bool
    """
    if not Config._main:
      Config.main()
    return Config._is_train

  @staticmethod
  def setup_logger(log_level=logging.DEBUG):
    """
    Logger Configurator

    Configures the logger.

    :param log_level: Level to log
    :type log_level: int
    """
    data_format = '%m/%d/%Y %I:%M:%S %p'  # Example Time Format - 12/12/2010 11:46:36 AM

    period_loc = Config.main().rfind(".")
    filename = Config.main()[:period_loc] + ".log"
    logging.basicConfig(filename=filename, level=log_level,
                        format='%(asctime)s -- %(message)s', datefmt=data_format)

    # Also print to stdout
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.info("************************ New Run Beginning ************************")


def _pickle_export(obj, filename):
  """
  Pickle Exporter

  Pickles the specified object and writes it to the specified file name.

  :param obj: Object to be pickled.
  :type obj: Object

  :param filename: File to write the specified object to.
  :type filename: str
  """
  try:
    os.makedirs(os.path.dirname(filename))
  except FileExistsError:
    pass
  with open(filename, 'wb') as f:
    pickle.dump(obj, f)


def _pickle_import(filename):
  """
  Pickle Importer

  Helper function for importing pickled objects.

  :param filename: Name and path to the pickle file.
  :type filename: str

  :return: The pickled object
  :rtype: Object
  """
  with open(filename, 'rb') as f:
    obj = pickle.load(f)
  return obj
