import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from src.utils.seaborn.legend_utils import format_legend

def plot_metrics(result, metrics=None, save_path=None):
    """
    :param result: an instance of src.eval.results.PartialResult
    """
    plt.figure()
    sns.set(rc={'figure.figsize':(10,7)})
    pd = result.to_pandas(metrics=metrics)
    ax = sns.lineplot(data=pd, x='Epoch', y='Value', style='Split', hue='Metric')
    format_legend(ax, ['Metric', 'Split'])
    if save_path:
        plt.savefig(os.path.join(save_path, result.name+'.png'), 
            facecolor='w', bbox_inches="tight", dpi = 300)
