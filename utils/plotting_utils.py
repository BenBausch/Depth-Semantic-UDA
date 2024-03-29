# ------------------------------------------------------------------------------
# Simple functions used for plotting
# ------------------------------------------------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt

def change_width_of_seaborn_bar_plot(ax, new_value):
    """
    Copied from
    https://stackoverflow.com/questions/34888058/changing-width-of-bars-in-bar-chart-created-using-seaborn-factorplot
    """
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
