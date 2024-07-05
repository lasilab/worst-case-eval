'''
High-level running of the folktables_copy_parallel script for all models.
'''

import multiprocessing
import subprocess
import pickle
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import argparse
from tqdm import tqdm

def run_single_experiment(cmd):
    subprocess.run(cmd.split(" "))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_train", action="store_false") # default to training
    args = parser.parse_args()
    train = args.no_train
    print(f"Training Models: {train}")

    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"][:5]
    
    # make sure all data downloaded
    run_single_experiment("python download_states.py")

    start = time.perf_counter()
    if not os.path.exists("final_results"):
        os.makedirs("final_results")
    count = multiprocessing.cpu_count()

    tasks = ["income"]
    train_files = ["folktables_epo.py", "folktables_notepo.py"]
    train_methods = ['SPO', 'CE']

    # train models
    if train:
        run_single_experiment("python train_models.py")

    print("{} cores available".format(count))

    train_time = time.perf_counter()
    print("trained models in {}".format((train_time - start) / 60))
    
    # for each of the datasets, parallelize over that dataset
    income_model_names = []

    for state in states:
        for method in train_methods:
            income_model_names.append("{}_{}_income".format(state, method))
    
    fns = "knapsack.top-k.util.fair.mse"

    model_str1 = ".".join(income_model_names)
    numdata = 25
    sample_num = 10000

    run_single_experiment(f"python folktables_copy_parallel.py {numdata} {sample_num} {model_str1} fw {fns} {numdata} income")

    optimize_time = time.perf_counter()

    print("optimized models in {}".format((optimize_time - start) / 60))

    eval_time = time.perf_counter()
    print("evaluated models in {}".format((eval_time - start) / 60))
    
    loss_names = fns.split('.')

    # analyze the results
    for task in tqdm(tasks):
        for method in train_methods:
            agg_results = np.array([[0. for _ in range(len(loss_names))] for _ in range(len(loss_names))])
            for state in states:
                with open('{}_{}_{}_optimized_{}_{}_{}/final_results.pickle'.format(state, method, task, sample_num, numdata, numdata), 'rb') as handle:
                    agg_results += (pickle.load(handle) / len(states))
            with open("final_results/{}_{}_agg.pickle".format(method, task), 'wb') as handle:
                pickle.dump(agg_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

            plt.matshow(agg_results)
            cb = plt.colorbar()
            plt.title("Aggregate Results for Dataset {}; Model {}".format(task, method))
            ticks = [i for i in range(len(loss_names))]
            plt.xticks(ticks, loss_names, rotation=20)
            plt.yticks(ticks, loss_names, rotation=20)
            plt.xlabel("Loss Function")
            plt.ylabel("Distribution")
            plt.savefig("final_results/{}_{}_agg.png".format(method, task), bbox_inches='tight')
    end = time.perf_counter()
    print("finished everything in {} min".format((end - start) / 60))
