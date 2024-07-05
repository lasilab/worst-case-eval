'''
Generate final efficiency plots over all datasets, comparing our method to pyomo/ipopt
'''

from matplotlib import pyplot as plt
import pickle
import gc
from collections import defaultdict
import os

plt.rc("axes",titlesize=20)
plt.rc("axes",labelsize=20)
plt.rc("font",size=14)

if __name__ == "__main__":
    folders = ["binary_final_results", "binary_final_results", "regression_final_results"]
    tasks = ["SPO", "CE"]
    sample_nums = [10,100,300,1000,1500,2000,3000]
    final_pcts = defaultdict(list)

    color_map = {
        "skim": "#CC79A7",
        "top-k": "#CC79A7",
        "acc": "#D55E00",
        "knapsack": "#0072B2",
        "fair": "#F0E442",
        "util": "#009E73",
        "ce": "#56B4E9",
        "mse": "#E69F00"
    }

    # go through each experiment folder and plot efficiency results on top of each other
    for i, folder in enumerate(folders):
        loss_fns = ['skim', 'acc', 'knapsack', 'fair', 'ce'] if i != 2 else ['knapsack', 'top-k', 'util', 'fair', 'mse']
        setting = "income" if i != 0 else "employment"
        with open(f"{folder}/agg_results_grid_{setting}.pickle", 'rb') as handle:
            agg_results_grid = pickle.load(handle)
        with open(f"{folder}/obj_vals_grid_{setting}.pickle", 'rb') as handle:
            final_objs_grid = pickle.load(handle)
        for task in tasks:
            for loss_fn in loss_fns:
                ests_wrt_sample_size = agg_results_grid[task][loss_fn]
                ref = final_objs_grid[task][loss_fn]
                if loss_fn != "ce": 
                    ests_wrt_sample_size = [x / ref for x in ests_wrt_sample_size]
                else: ests_wrt_sample_size = [ref / x for x in ests_wrt_sample_size]

                final_val = ests_wrt_sample_size[-1]

                plt.plot(sample_nums, ests_wrt_sample_size, c=color_map[loss_fn])

    for loss_fn in color_map:
        if loss_fn == "skim": continue
        plt.plot(sample_nums, [1 for _ in range(len(sample_nums))], label=loss_fn, c=color_map[loss_fn])
    plt.plot(sample_nums, [1 for _ in range(len(sample_nums))], label="optimal", linewidth=2.5, c="black")
    plt.legend()
    plt.xlabel("Number samples per iter FW")
    plt.ylabel("Converged Loss of Model")
    plt.tight_layout()

    if not os.path.exists("efficiency_experiments"):
        os.makedirs("efficiency_experiments")

    plt.savefig("efficiency_experiments/final_vis.pdf", format="pdf", bbox_inches="tight")
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')   
    gc.collect()