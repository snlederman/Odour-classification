"""
plots the training history
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_training_df(history, metric):
    training_df = pd.DataFrame({"epochs" : range(1,len(history.history[metric])+1), "train" : history.history[metric], "validation" : history.history[f"val_{metric}"]})
    training_df.set_index("epochs", inplace=True)
    return training_df

def plot_history(history, metric, output_file):
    training_df = get_training_df(history, metric)
    training_df.plot()
    if metric == "loss":
        best_val = training_df["validation"].argmin() # epoch of minimum lass in validation 
    else:
        best_val = training_df["validation"].argmax() # epoch of maximum accuracy in validation 
    plt.plot(
        [best_val + 1]*len(training_df), # X
        np.linspace(training_df[["train","validation"]].min().min(), training_df[["train","validation"]].max().max(), len(training_df)), # Y
        linestyle="dashed",
        label="best epoch")
    plt.legend()
    plt.ylabel(metric)
    plt.xlabel("epochs")
    plt.title(f"{metric} history")
    plt.savefig(output_file)