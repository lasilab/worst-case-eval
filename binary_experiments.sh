#!/usr/bin/env bash

cd binary_final_results
python folktables_batchrunner.py --no_train # if you want to train your own models, remove
python folktables_polynomial_batch.py # runs experiment comparison to pyomo/ipopt
python eda.py # visualize, for each worst-case distribution, converged weights and model predictions for individuals in the optimization instance