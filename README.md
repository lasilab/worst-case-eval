# Decision-Focused Evaluation of Worst-Case Distribution Shift

## Overview

This repository is the official implementation of the paper "Decision-Focused Evaluation of Worst-Case Distribution Shift" (UAI 2024). If you find this repository useful or use this code in your research, please cite the following paper:

[TODO add bibtex citation]

_Note_: To run a simplified version of our experiments (5 states, 25 individuals per optimization instance, 10,000 samples per iteration of the Frank-Wolfe algorithm) please switch to the `mini` branch via the following command

```
git checkout origin/mini
```

## Installing Dependencies

<!-- To install requirements, setup a conda environment using the following command: -->
To install requirements, run the following commands:

```
source install.sh
conda activate SPODR
```

## Experiments

<!-- ### Predictive Model Training -->

### Setup 

To use the provided model checkpoints, unzip them in the proper directories using the following command. To train the models from scratch instead, skip this step.

```
./move_models.sh
```

This can also be done manually:

- Move the `binary_final_results_models.zip` into the `binary_final_results` folder and unzip
- Move the `regression_final_results_models.zip` into the `regression_final_results` folder and unzip

To train the models from scratch, comment out `--no_train` in the `binary_experiments.sh` and `regression_experiments.sh` files. 

<!-- *If training from scratch, do **not** run* `move_models.sh`. -->

<!-- To replicate our experiments, all trained predictive models must be placed and unzipped in the proper directories. This can either be done by training them directly by commenting out `--no_train` in the `binary_experiments.sh` and `regression_experiments.sh` files, or using the provided zip files in the root directory. To exercise the second option, run the procedure below. Otherwise (training predictive models from scratch), skip this step.

- move the `binary_final_results_models.zip` into the `binary_final_results` folder and unzip
- move the `regression_final_results_models.zip` into the `regression_final_results` folder and unzip

The process of reusing the pretrained predictive models can also be accomplished by running:

```
./move_models.sh
``` -->

### Quick Experiments

To run all experiments and reproduce our results (Figures 1, 2, 3, 4), run the following command. The results will be placed inside folders `_final_results_ci`, `efficiency_experiments`, and `paper_visualizations` under the root directory. 
<!-- (To run experiments _individually_, skip to the "Running Individual Experiments" section below).  -->

<!-- run all experiments on the binary prediction tasks (unemployment, income classification) and income regression task (identify worst-case distributions w.r.t. all loss functions, for all predictive models, compile the results, and then compare our method to Pyomo/IPOPT): -->

```
./run_all.sh
```

<!-- 
This contains visualizations (for each worst-case distribution) of the model predictions and converged weights assigned to individuals within the corresponding optimization instance. -->


<!-- To obtain final results with confidence intervals, along with the results of an efficiency-related experiment comparing our method to Pyomo/IPOPT, run the following command. The results will be placed inside folders `_final_results_ci`, `efficiency_experiments`, and `paper_visualizations` under the root directory. This contains visualizations (for each worst-case distribution) of the model predictions and converged weights assigned to individuals within the corresponding optimization instance (TODO: list figures). -->

<!-- ```
./final_experiments.sh
``` -->


<!-- To run all low-level experiments (identifying worst-case distributions w.r.t. all combinations of predictive model, optimization instance, and metric, along with Pyomo/IPOPT comparison), run the following command. (To run experiments _individually_, skip to the "Running Individual Experiments" section below; note that _only one of the Quick Experiments and Individual Experiments Sections should be run, not both_). -->

<!-- ### Experiment Running

After the above scripts have been run, the following command will run all experiments and output the results inside the folders `_final_results_ci`, `efficiency_experiments`, and `paper_visualizations` under the root directory: -->


<!-- ### Running Individual Experiments -->

<!-- Should you decide to run experiments one-at-a-time *as opposed* to simply running run_all.sh, run the following commands. The provided commands (which can be run in any order) will run all experiments on the binary prediction tasks (unemployment, income classification) and income regression task (identify worst-case distributions w.r.t. all loss functions, for all predictive models, compile the results, and then compare our method to Pyomo/IPOPT): -->
<!-- Should you decide to run experiments one-at-a-time, run the following commands. The provided commands can be run in any order:


```
./binary_experiments.sh
./regression_experiments.sh
``` -->
<!-- 
### High-Level Experiments
Once either `run_all.sh` or a combination of `binary_experiments.sh` and `regression_experiments.sh` have successfully run, the following command to obtain final results with confidence intervals, along with the results of an efficiency-related experiment comparing our method to Pyomo/IPOPT. The results will be placed inside folders `_final_results_ci`, `efficiency_experiments`, and `paper_visualizations` under the root directory: -->



<!-- The final diagrams seen in the paper should be located within the folders `_final_results_ci`, `efficiency_experiments`, and `paper_visualizations` (this contains visualizations, for each worst-case distribution, of the model predictions and converged weights assigned to individuals within the corresponding optimization instance). -->

## License 

This repository is licensed under the terms of the [MIT License](https://github.com/kren333/worst_case_resource_allocation_final?tab=MIT-1-ov-file).

## Questions?

For more details, refer to the accompanying paper: [Decision-Focused Evaluation of Worst-Case Distribution Shift](TODO: Insert arxiv link). 
If you have questions, please feel free to reach us at kevinren@andrew.cmu.edu or to open an issue.

