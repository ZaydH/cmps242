from enum import Enum
import argparse
import __main__
import os


class DecisionFunction(Enum):
  ArgMax = 0


class Config(object):

  model_dir = "./model/"

  WINDOW_SIZE = 50
  EMBEDDING_SIZE = 30
  RNN_HIDDEN_SIZE = 64

  class Train(object):
    input_file = "." + os.sep + "trump_speeches.txt"
    epochs = 100
    batch_size = 100

  class FF(object):
    depth = 1
    hidden_width = 128

  class DecisionEngine(object):
    """
    Configuration settings for the decision engine.
    """
    function = DecisionFunction.ArgMax

  @staticmethod
  def parse_args():
    # Select the arguments based on what program is running
    sep_loc = __main__.__file__.rfind(os.sep)
    main_file_name = __main__.__file__[sep_loc + 1:]
    if main_file_name == "train.py":
      Config._train_args()
    elif main_file_name == "trump.py":
      Config._trump_args()
    else:
      raise ValueError("Unknown main file.")

  @staticmethod
  def _train_args():
    parser = argparse.ArgumentParser("Character-Level RNN Trainer")
    parser.add_argument("--train", type=str, required=False,
                        default=Config.Train.input_file,
                        help="Path to the training set file.")
    parser.add_argument("--model", type=str, required=False,
                        default=Config.model_dir,
                        help="Directory to which to export the trained network")
    parser.add_argument("--seq-len", type=int, required=False,
                        default=Config.WINDOW_SIZE,
                        help="RNN sequence length")
    parser.add_argument("--epochs", type=int, required=False,
                        default=Config.Train.epochs,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, required=False,
                        default=Config.Train.batch_size,
                        help="Batch size")
    args = parser.parse_args()

    Config.Train.input_file = args.train
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
