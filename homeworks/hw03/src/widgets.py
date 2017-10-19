from IPython.display import Javascript, display
import ipywidgets
import const


def run_all(_):
  # noinspection PyTypeChecker
  display(Javascript('IPython.notebook.execute_cells_below()'))


k_slider = ipywidgets.IntSlider(
  value=10,
  min=2,
  max=20,
  step=1,
  disabled=False,
  continuous_update=False,
  orientation='horizontal',
  readout=True,
  readout_format='d',
  width=1000,
)
k_hbox = ipywidgets.HBox([ipywidgets.Label('Number of Folds: '), k_slider])


learning_alg_radio = ipywidgets.RadioButtons(
  options=[const.ALG_GD, const.ALG_EG],
  description="",
  disabled=False
)
learning_alg_radio.value = const.ALG_EG
learning_alg_hbox = ipywidgets.HBox([ipywidgets.Label("Select Learning Algorithm: "),
                                     learning_alg_radio])


regularizer_radio = ipywidgets.RadioButtons(
  options=[const.REGULARIZER_L1_NORM, const.REGULARIZER_L2_NORM],
  description="",
  disabled=False
)
regularizer_radio.value = const.REGULARIZER_L2_NORM
regularizer_hbox = ipywidgets.HBox([ipywidgets.Label("Select the Regularizer: "),
                                    regularizer_radio])


error_type_radio = ipywidgets.RadioButtons(
  options=[const.ERROR_ACCURACY, const.ERROR_RMS],
  description="",
  disabled=False
)
error_type_radio.value = const.ERROR_ACCURACY
error_type_hbox = ipywidgets.HBox([ipywidgets.Label("Select the Validation Error Calculation: "),
                                   error_type_radio])


run_button = ipywidgets.Button(
  description='Run Learner',
  disabled=False,
  button_style='',  # 'success', 'info', 'warning', 'danger' or ''
  tooltip='Run Learning Algorithm with the specified paramters',
  icon='check'
)
run_button.on_click(run_all)


learning_rate_slider = ipywidgets.FloatSlider(
  value=20,
  min=0.1,
  max=100,
  step=0.1,
  orientation='horizontal',
  readout=True,
  readout_format='.2f',
)
learning_rate_hbox = ipywidgets.HBox([ipywidgets.Label("Learning Rate ($\eta$): "),
                                      learning_rate_slider])


lambdas_range_slider = ipywidgets.IntRangeSlider(
  value=[-10, 10],
  min=-10,
  max=10,
  step=1,
  orientation='horizontal',
  readout=True,
  readout_format='d',
)
lambdas_range_hbox = ipywidgets.HBox([ipywidgets.Label("Range of $\lambda$ in Form $2^{x}$: "),
                                      lambdas_range_slider])


update_results_button = ipywidgets.Button(
  description='Update Results',
  disabled=False,
  button_style='',  # 'success', 'info', 'warning', 'danger' or ''
  tooltip='Update the table and graph',
  icon='check'
)
update_results_button.on_click(run_all)
