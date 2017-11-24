import enum
import argparse
import __main__

class DecisionFunction(enum):
  ArgMax = 0


class Config(object):

  model_dir = "./model/"

  WINDOW_SIZE = 50
  EMBEDDING_SIZE = 30
  RNN_HIDDEN_SIZE = 64

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
    if __main__.__file__ == "train.py":
      Config._train_args()
    elif __main__.__file__ == "trump.py":
      Config._trump_args()
    else:
      raise ValueError("Unknown main file.")

  @staticmethod
  def _train_args():
    parser = argparse.ArgumentParser("Character-Level RNN Trainer")
    parser.add_argument("--train", type=str, required=False,
                        default="trump_speeches.txt",
                        desc="Path to the training set file.")
    parser.add_argument("--model", type=str, required=False,
                        default=Config.model_dir,
                        desc="Directory to which to export the trained network")
    parser.add_argument("--seq-len", type=int, required=False,
                        default=Config.WINDOW_SIZE,
                        desc="RNN sequence length")

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
