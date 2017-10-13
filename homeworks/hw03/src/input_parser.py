import nltk
import pandas as pd
import string  # Used for the puncation class

def parse_input_data():
  train_data = parse_file("test.csv")
  test_data = parse_file("train.csv")


def remove_punctuation(s):
  """
  String Punctuation Remover

  Helper function used in an "apply" Pandas method to clean strings strings with punctuation.

  :param s: String
  :type s: str
  :return: String with all punc
  :rtype: str
  """
  s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
  return s


def parse_csv_data_file(csv_file_path):
  """
  HW03 Data File Parser

  Parses homework #3 data file.  It does string cleaning including making everything lower case, removing
  unparseable chacters, removing stop words, removing punctuation, and tokenizing.

  :param csv_file_path: Path to a data file
  :type csv_file_path: str

  :return: Pandas data frame containing tokenized entries.
  :rtype: pd.Dataframe
  """
  # Build the data as a pandas dataframe
  col_names = ['label', 'sms', 'sms1', 'sms2', 'sms3']
  df = pd.read_csv(csv_file_path, names=col_names, header=0, dtype=object)
  df.fillna("", inplace=True)
  df[col_names[1]] = df[col_names[1:]].apply(lambda x: '{} {} {} {}'.format(x[0], x[1], x[2], x[3]), axis=1)
  df.drop(labels=col_names[2:], axis=1, inplace=True)

  # Perform string cleaning
  stop_words = nltk.corpus.stopwords.words('english')
  df[col_names[1]] = (df[col_names[1]].str.lower()  # All lower case
                                      .str.replace('[^\x00-\x7F]', '')  # Re-encode the string to remove illegal chars
                                      .apply(remove_punctuation)
                                      .apply(nltk.word_tokenize)
                                      .apply(lambda x: [item for item in x if item not in stop_words])) # Remove stop words
  return df


if __name__ == "__main__":
  parse_input_data()
