import pandas as pd
import numpy as np
from run_learner import build_lambdas
import math


def _highlight_min(s):
  is_min = s == s.min()
  # noinspection PyTypeChecker
  return ['background-color: yellow; font-weight: bold;' if v else '' for v in is_min]


def get_lambda_str(x):
  """
  Builds the string for the lambda in the table.  Reesult uses LaTeX formatting.

  :param x: Power of two for latex
  :type x: int
  :return:
  :rtype: str
  """
  return "$2^{%d}$" % (round(math.log(x, 2))) if x != 0 else str(x)


def create_table(train_err, validation_err, test_err):
  """
  Pandas Table Creator

  :return: Results combined into a single DataFrame
  :rtype: pd.DataFrame
  """
  # Create the lambdas DataFrame
  lambdas_str = [get_lambda_str(x) for x in build_lambdas()]
  df = pd.DataFrame({'$\lambda$': lambdas_str})

  # Create the errors DataFrame
  column_names = ["Training", "Validation", "Test"]
  data_arr = np.stack((train_err[:, 0], validation_err[:, 0], test_err[:, 0]), axis=-1)
  df_errors = pd.DataFrame(data_arr, columns=column_names)

  # Merge the DataFrames
  df = pd.concat([df, df_errors], axis=1)

  # noinspection PyUnresolvedReferences
  return df.transpose()


def stylize_table(results_table):
  """
  Stylizes the results table.

  :return: Stylized Pandas table.
  :rtype: pd.io.formats.style.Styler
  """
  # Format the Pandas table for printing in Jupyter notebook.
  column_names = ["Training", "Validation", "Test"]
  th_styles = [
    dict(selector=".row_heading", props=[('color', 'black'),  # Ensure the headers have proper format
                                         ('border-color', 'black'),
                                         ('border-style', 'solid'),
                                         ('border-width', '1px'),
                                         ('text-align', 'center')]),
    dict(selector=".col_heading", props=[('display', 'none')]),  # Hide the index row.
  ]
  # noinspection PyUnresolvedReferences
  results_styler = (results_table.style.apply(_highlight_min, axis=1, subset=pd.IndexSlice[column_names, :])
                                       .set_properties(**{'color': 'black',
                                                          'border-color': 'black',
                                                          'border-style': 'solid',
                                                          'border-width': '1px',
                                                          'text-align': 'center'})
                                       .set_table_styles(th_styles)
                                       .format("{:.4f}", subset=pd.IndexSlice[column_names, :]))
  return results_styler


if __name__ == "__main__":
  """Debug only code."""
  # noinspection PyProtectedMember
  from run_learner import _build_random_results
  training, validation, test = _build_random_results()
  create_table(training, validation, test)
