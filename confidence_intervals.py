'''
Generate high-level grid results, along with 95% confidence intervals
'''

import numpy as np
import pickle
from matplotlib import pyplot as plt
import gc
import os

plt.rc("axes",titlesize=20)
plt.rc("axes",labelsize=20)
plt.rc("font",size=13)

if __name__ == "__main__":
    mainpaths = ["binary_final_results", "regression_final_results"]

    for mainpath in mainpaths:
        states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
        tasks = ["employment", "income"] if "binary" in mainpath else ["income"]
        methods = ["SPO", "CE"]
        loss_fns = ["knapsack", "top-k", "acc", "fair", "ce"] if "binary" in mainpath else ["knapsack", "top-k", "util", "fair", "mse"]

        # compile and diagonal-normalize each final results grid, with the confidence intervals
        for task in tasks:
            for method in methods:
                samples = [[[] for _ in range(len(loss_fns))] for _ in range(len(loss_fns))]
                bounds = [[None for _ in range(len(loss_fns))] for _ in range(len(loss_fns))]
                for model_state in states:
                    model_path = f"{model_state}_{method}_{task}"
                    with open(f"{mainpath}/{model_path}_optimized_35000_40_40/final_results.pickle", 'rb') as handle:
                        grid = pickle.load(handle)
                    assert(grid.shape == (5,5))

                    if "binary" in mainpath:
                        for i in range(4):
                            grid[:,i] /= grid[i,i]
                        grid[:,-1] -= 1
                        grid[:,-1] = grid[-1,-1] / grid[:,-1]
                        # arr = np.around(grid, decimals=3)

                        # switch around columns: knapsack, top-k, fair, acc, ce
                        grid[:, [2, 0]] = grid[:, [0, 2]]
                        grid[:, [2, 3]] = grid[:, [3, 2]]
                        # switch around rows, the same way
                        grid[[2, 0],:] = grid[[0, 2],:]
                        grid[[2, 3],:] = grid[[3, 2],:]
                    
                    else:
                        for i in range(5):
                            grid[:,i] /= grid[i,i]
                    
                    for i in range(5):
                        for j in range(5):
                            samples[i][j].append(grid[i,j])

                for i in range(5):
                    for j in range(5):
                        # assert(len(samples[i][j]) == 50)
                        sd = np.std(samples[i][j])
                        sd /= np.sqrt(len(samples[i][j]))
                        bounds[i][j] = sd * 1.96
                
                bounds_change = np.array(bounds)

                final_results_grid = np.mean(np.array(samples), axis=2) # take average over all entries in 5x5 grid

                fig, ax = plt.subplots()

                ax.matshow(final_results_grid)

                for i in range(len(final_results_grid)):
                    for j in range(len(final_results_grid)):
                        c = final_results_grid[j,i]
                        ax.text(i, j, f"{np.around(c, decimals=3)}\n±{np.around(bounds_change[j,i], decimals=3)}", va='center', ha='center')

                ticks = [i for i in range(len(loss_fns))]
                plt.xticks(ticks, loss_fns, rotation=20)
                plt.yticks(ticks, loss_fns, rotation=20)
                plt.xlabel("Eval Metric")
                plt.ylabel("Distribution")
                # plt.title("Aggregate Results for Dataset {}; Model {}".format(task, method))
                plt.tight_layout()
                dataset = mainpath.split("_")[0]

                if not os.path.exists("_final_results_ci"):
                    os.makedirs("_final_results_ci")

                plt.savefig(f"_final_results_ci/{dataset}_{method}_{task}_bounds.pdf", format="pdf", bbox_inches='tight')
                plt.close('all')
                gc.collect()