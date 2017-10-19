import nltk
import pandas as pd
import numpy as np
import string  # Used for the punctuation class
import sklearn.feature_extraction.text as sklearntext
import const


def _print_data_sizes(train_data, test_data):
  """
  Prints Data Sizes

  This function is used as a helper to print the training data sizes.
  It is just used for making the Jupyter notebook more readable.

  :param train_data: Training data set.
  :type train_data: pd.DataFrame
  :param test_data: Test data set.
  :type test_data: pd.DataFrame
  """
  print ("Number of Training Samples: " + str(train_data.shape[0])
         + "\nNumber of Test Samples: " + str(test_data.shape[0]))


def parse(override_input_data=False):
  """
  SMS Data Parser

  Parses the test and training sets.

  :return: Parsed training and test data respectively.
  :rtype: Tuple(pd.DataFrame)
  """
  train_data = parse_csv_data_file("train.csv")
  test_data = parse_csv_data_file("test.csv")
  if override_input_data:
    train_data = parse_csv_data_file("coolest_guy_zayd_train.csv")
    test_data = parse_csv_data_file("coolest_guy_zayd_test.csv")

  # Vectorizer and get the TF-IDF scores
  data_col_name = "sms"
  train_vectorize, train_vocab = count_vectorizer(train_data, data_col_name)
  test_vectorize, _ = count_vectorizer(test_data, data_col_name, train_vocab)

  # convert the target labels to 0/1 for learning ease.
  label_col_name = "label"
  configure_targets(train_data, label_col_name)
  configure_targets(test_data, label_col_name)

  # Helper function for the reader.
  _print_data_sizes(train_data, test_data)

  # Store the results in a dictionaries
  train_target = np.matrix(train_data[const.target].values).transpose()
  test_target = np.matrix(test_data[const.target].values).transpose()
  return (np.hstack([train_target, train_vectorize]),
          np.hstack([test_target, test_vectorize]))


def configure_targets(df, col_name):
  """
  Standardizes the format for ham/spam in the data frame.

  :param df: Training or test data including target label
  :type df: pd.DataFrame
  :param col_name: Target column name in the Pandas DataFrame
  :type col_name: str
  """
  df[const.target] = df[col_name].apply(lambda x: const.HAM if str(x).lower() == "ham" else const.SPAM)


def count_vectorizer(df, col_name, vocab=None):
  """
  String Vectorizer With Optional Dictionary Support

  Given a Pandas DataFrame, this function will tokenizer using either the provided dictionary
  or the one implicit to the data itself.  It then returns a matrix of the tf-idf
  scores of each of the words.

  :param df: Source data frame to vectorize
  :type df: pd.DataFrame
  :param col_name: Name of the feature column in the pandas DataFrame
  :type col_name: string
  :param vocab: Dictionary of support dictionary words to mapping of index number
  :type vocab: dict
  :return: TF-IDF word matrix and vocabulary.
  :rtype: Tuple(pd.DataFrame, dict)
  """
  stop_words = nltk.corpus.stopwords.words('english')
  vectorizer = sklearntext.CountVectorizer(lowercase=True, stop_words=stop_words, vocabulary=vocab)
  doc_word_matrix = vectorizer.fit_transform(df[col_name])
  if vocab is None:
    vocab = vectorizer.vocabulary_

  tf_idf = sklearntext.TfidfTransformer(norm=None).fit_transform(doc_word_matrix)
  return tf_idf.toarray(), vocab


def remove_punctuation(s):
  """
  String Punctuation Remover

  Helper function used in an "apply" Pandas method to clean strings strings with punctuation.

  :param s: String
  :type s: str
  :return: String with all punctuation removed.
  :rtype: str
  """
  s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
  return s


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
  col_names = ['label', 'sms']
  df = pd.read_csv(csv_file_path, names=col_names, header=0, dtype=object)

  # Perform string cleaning
  df[col_names[1]] = (df[col_names[1]].str.replace('[^\x00-\x7F]', '')  # Re-encode the string to remove illegal chars
                                      .apply(remove_punctuation))  # Remove stop words
  return df


if __name__ == "__main__":
  parse()
