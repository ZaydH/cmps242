import nltk
import pandas as pd
import string  # Used for the punctuation class
import sklearn.feature_extraction.text as sklearntext


def parse():
  train_data = parse_csv_data_file("train.csv")
  test_data = parse_csv_data_file("test.csv")

  # Vectorizer and get the TF-IDF scores
  data_col_name = "sms"
  train_data[data_col_name + "_tfidf"], train_vocab = count_vectorizer(train_data, data_col_name)
  test_data[data_col_name + "_tfidf"], _ = count_vectorizer(test_data, data_col_name, train_vocab)

  return train_data, test_data


def count_vectorizer(df, col_name, vocab=None):
  """
  String Vectorizer With Optional Dictionary Support

  Given a Pandas dataframe, this function will tokenizer using either the provided dictionary
  or the one implicit to the data itself.  It then returns a matrix of the tf-idf
  scores of each of the words.

  :param df: Source data frame to vectorize
  :type df: pd.Dataframe
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

  tf_idf = sklearntext.TfidfTransformer().fit_transform(doc_word_matrix)
  return tf_idf, vocab


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
  # Build the data as a pandas dataframe
  col_names = ['label', 'sms', 'sms1', 'sms2', 'sms3']
  df = pd.read_csv(csv_file_path, names=col_names, header=0, dtype=object)
  df.fillna("", inplace=True)
  df[col_names[1]] = df[col_names[1:]].apply(lambda x: '{} {} {} {}'.format(x[0], x[1], x[2], x[3]), axis=1)
  df.drop(labels=col_names[2:], axis=1, inplace=True)

  # Perform string cleaning
  df[col_names[1]] = (df[col_names[1]].str.replace('[^\x00-\x7F]', '')  # Re-encode the string to remove illegal chars
                                      .apply(remove_punctuation))  # Remove stop words
  return df


if __name__ == "__main__":
  parse()
