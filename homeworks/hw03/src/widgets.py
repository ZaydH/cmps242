from IPython.display import Javascript, display
import ipywidgets
from run_learner import *
import const

k_widget = ipywidgets.IntSlider(
    value=10,
    min=2,
    max=20,
    step=1,
    description='Number of Folds:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

learning_alg_radio = ipywidgets.RadioButtons(
    options=[const.GD_ALG, const.EG_ALG],
    description='Select Learning Algorithm: ',
    disabled=False
)

run_button = ipywidgets.Button(
    description='Run Learner',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Run Learning Algorithm with the specified paramters',
    icon='check'
)

def run_all(ev):
    display(Javascript('IPython.notebook.execute_cells_below()'))
run_button.on_click(run_all)
