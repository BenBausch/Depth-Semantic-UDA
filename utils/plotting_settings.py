# This file defines the setting for plotting in order to create consistent looking plots

# dependencies
import matplotlib.pyplot as plt
import seaborn as sns


def set_plotting_settings():
    """Fix plotting values to make consistent plots. Call this function at the beginning of your script."""
    sns.set()
    sns.set_style("darkgrid")
