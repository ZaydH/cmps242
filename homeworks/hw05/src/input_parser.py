import string
import sklearn.feature_extraction.text as sklearntext
import pandas as pd
from mlxtend.preprocessing import one_hot
import numpy as np
import const
import gensim

# load the word2vec model from google
w2v = None

def parse():
  """
  Email One Hot Parser

  Parses the test and training sets.

  :return: Parsed training and test data respectively.
  :rtype: Tuple(pd.DataFrame, pd.DataFrame, dict)
  """
  train = parse_csv_data_file("train.csv")
  test = parse_csv_data_file("test.csv")

  # load the word2vec model from google
  global w2v
  w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

  full_vocab = build_vocab(train, test)

  build_integer_token_representation(train, test, full_vocab)

  build_one_hot(train, test, full_vocab)

  build_word_vectors(train,test,full_vocab)

  dummy_word = w2v.word_vec("poop")
  w2v = None

  return train, test, full_vocab, dummy_word


def build_vocab(train_data, test_data):
  """
  Dataset Dictionary Builder

  Vectorizers and builds the combined dictionary of the test and training datasets.

  :param train_data: DataFrame of the training data
  :type train_data: pd.DataFrame

  :param test_data: DataFrame of the test data
  :type test_data: pd.DataFrame

  :return: Combined vocabulary for the training and testing.
  :rtype: dict
  """
  # Token pattern is used to allow the vectorizer to include one letter words (e.g., "a", "i")
  vectorizer = sklearntext.CountVectorizer(lowercase=True, token_pattern=r"[\b|\w*#]*\w*[!+#*\w*|\b]*")
  words = set()
  for data_pd in [train_data, test_data]:
    vectorizer.fit_transform(data_pd[const.COL_TWEET])
    vocab = vectorizer.vocabulary_
    words |= set(vocab.keys())
  vocab = {}
  for idx, word in enumerate(sorted(words)):
    vocab[word] = idx
  return vocab


def build_integer_token_representation(train_data, test_data, vocab):
  """
  Converts the entire tweet message into an integer representation with respect
  to the specified vocab.

  :param train_data: Training data set
  :type train_data: pd.DataFrame

  :param test_data: Test data set
  :type test_data: pd.DataFrame

  :param vocab: Complete vocabulary for the training and test data.
  :type vocab: dict
  """
  f = lambda df: [[vocab[word] for word in tweet.split()] for tweet in df[const.COL_TWEET]]
  train_data[const.COL_TWEET_INT_TRANSFORM] = f(train_data)
  test_data[const.COL_TWEET_INT_TRANSFORM] = f(test_data)

  f = lambda df: [[word for word in tweet.split()] for tweet in df[const.COL_TWEET]]
  train_data[const.COL_TWEET_WORD_TRANSFORM] = f(train_data)
  test_data[const.COL_TWEET_WORD_TRANSFORM] = f(test_data)


def build_one_hot(train_data, test_data, vocab):
  """
  One-Hot Representation Builder

  Converts the integer tokenized representation into a one hot representation.

  :param train_data: Training data set
  :type train_data: pd.DataFrame
  :param test_data: Test data set
  :type test_data: pd.DataFrame
  :param vocab: Full vocabulary for the training and test sets
  :type vocab: dict
  """
  for df in [train_data, test_data]:
    # one_hot = enc.transform(train_data[const.COL_TWEET_TRANSFORM])
    df[const.COL_ONE_HOT] = df[const.COL_TWEET_INT_TRANSFORM].apply(lambda x: one_hot(x, num_labels=len(vocab)))



def build_word2vec_list(word_list):
  vecs = []
  for word in word_list:
    try:
      vec = w2v.word_vec(word)
      vecs.append(vec)
    except KeyError:
      continue
  return vecs


def build_word_vectors(train_data, test_data, vocab):
  """

  :param train_data: Training data set
  :type train_data: pd.DataFrame
  :param test_data: Test data set
  :type test_data: pd.DataFrame
  :param vocab: Full vocabulary for the training and test sets
  :type vocab: dict
  """
  for df in [train_data, test_data]:
    # one_hot = enc.transform(train_data[const.COL_TWEET_TRANSFORM])
    df[const.COL_WORD2VEC] = df[const.COL_TWEET_WORD_TRANSFORM].apply(build_word2vec_list)

def _remove_punctuation(s):
  """
  String Punctuation Remover

  Helper function used in an "apply" Pandas method to clean strings strings with punctuation.

  It does NOT remove "#" and "!" since those may be relevant for

  :param s: String
  :type s: str
  :return: String with all punctuation removed.
  :rtype: str
  """
  s = ''.join([i for i in s if i not in (frozenset(string.punctuation) - {"#", "!"})])
  return s


def _determine_target_label(x):
  ret = const.LBL_DONALD_TRUMP if str(x).lower() == const.HDL_DONALD_TRUMP.lower() else const.LBL_HILLARY_CLINTON
  return ret


def parse_csv_data_file(csv_file_path):
  """
  HW03 Data File Parser

  Parses homework #3 data file.  It does string cleaning including making everything lower case, removing
  unparseable characters, removing stop words, removing punctuation, and tokenizing.

  :param csv_file_path: Path to a data file
  :type csv_file_path: str

  :return: Pandas data frame containing tokenized entries.
  :rtype: pd.DataFrame
  """
  # Build the data as a pandas DataFrame
  col_names = [const.COL_HANDLE, const.COL_TWEET]
  df = pd.read_csv(csv_file_path, names=col_names, header=0, dtype=object)

  # Perform string cleaning
  df[const.COL_TWEET] = (df[const.COL_TWEET].str.replace('[^\x00-\x7F]', '')  # Remove illegal chars
                                            .str.lower()  # Make the text lower case
                                                .apply(_remove_punctuation))  # Remove punctuation
  df[const.COL_TARGET] = df[const.COL_HANDLE].apply(_determine_target_label)
  return df


if __name__ == "__main__":
  parse()
