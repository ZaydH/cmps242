import pandas as pd
import numpy as np
from IPython.display import display, HTML


def _highlight_max(s):
  is_max = s == s.max()
  return ['background-color: yellow; font-weight: bold;' if v else '' for v in is_max]


def create_table():
  np.random.seed(24)
  column_names = ["$\lambda$", "Training", "Validation", "Testing"]
  df = pd.DataFrame({'A': np.linspace(1, 10, 10)})
  df = pd.concat([df, pd.DataFrame(np.random.randn(10, 4), columns=column_names)],
                 axis=1)
  df.iloc[0, 2] = np.nan
  # Format the Pandas table for printing in Jupyter notebook.
  df = (df.transpose().style.apply(_highlight_max, axis=1)
                            .format("{:.2f}")
                            .set_properties(**{'color': 'black',
                                               'border-color': 'black',
                                               'border-style': 'solid',
                                               'border-width': '1px'})
                            .set_table_styles( # Hackish way to prevent the top row appearing appearing
                                              [{'selector': '.col_heading',
                                                'props': [('display', 'none')]},
                                               {'selector': '.blank.level0',
                                                'props': [('display', 'none')]}])
        )
  return df


if __name__ == "__main__":
  """Debug only code."""
  
  create_table()