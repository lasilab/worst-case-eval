'''
given a list of model paths, the setting they were trained on, optimizer, number data to skim, and task to optimize wrt (skim, knapsack, acc),
finds worst-case distribution shift for each of the models, wrt each of the loss functions, wrt each of the subpopulations
'''

import time
from folktables import ACSDataSource, ACSEmployment, ACSIncome
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import random
from collections import Counter
import warnings
from matplotlib import pyplot as plt
import argparse
import os
import math
import torch
import json
from tqdm import tqdm
from torch import nn
from collections import defaultdict
import gc

plt.rc("axes",titlesize=20)
plt.rc("axes",labelsize=20)
plt.rc("font",size=14)
# from folktables_epo import LogisticRegressionEmployment, NeuralNetworkIncome

# multiprocessing libs
from multiprocessing import Pool
from functools import partial
from concurrent.futures import ProcessPoolExecutor

# imports for running pyepo
#import gurobipy as gp
#from gurobipy import GRB
#import numpy as np
#from pyepo.model.grb import optGrbModel

warnings.filterwarnings('ignore')

# prediction models
class LogisticRegressionEmployment(nn.Module):    
    # build the constructor
    def __init__(self, max_min, num_feat, task):
        super(LogisticRegressionEmployment, self).__init__()

        # create embeddings
        self.cat_cols = [i for i in range(2, 16)]
        self.embedding_size_dict = max_min 
        self.embedding_dim_dict= {key: 5 for key,val in self.embedding_size_dict.items()}

        embeddings = {}
        self.total_embed_dim = 0
        for col in self.cat_cols:
            num_embeddings = self.embedding_size_dict[col]
            embedding_dim = self.embedding_dim_dict[col]
            embeddings[str(col)] = nn.Embedding(num_embeddings, embedding_dim)
            self.total_embed_dim+= embedding_dim
        self.embeddings = nn.ModuleDict(embeddings)

        self.num_feat = num_feat

        self.task = task

        self.linear = torch.nn.Linear(self.num_feat - len(self.cat_cols) + self.total_embed_dim, 1)
        
    # make predictions
    def forward(self, x):
        embedded_vector = x[:,:,:2]
        for col in range(2, self.num_feat):
            # we know that this col is categorical; map each row to its embedding and concat
            embeddings = self.embeddings[str(col)](x[:,:,col].type(torch.int64))
            embedded_vector = torch.cat((embedded_vector, embeddings), dim=2)
        out = torch.squeeze(torch.sigmoid(self.linear(embedded_vector)), -1)
        return out

class NeuralNetworkIncome(nn.Module):    
    # build the constructor
    def __init__(self, max_min, num_feat, task):
        super().__init__()

        # create embeddings
        self.cat_cols = [1,3,4,6,7]
        self.embedding_size_dict = max_min
        self.embedding_dim_dict= {key: 5 for key,val in self.embedding_size_dict.items()}

        embeddings = {}
        self.total_embed_dim = 0
        for col in self.cat_cols:
            num_embeddings = self.embedding_size_dict[col]
            embedding_dim = self.embedding_dim_dict[col]
            embeddings[str(col)] = nn.Embedding(num_embeddings, embedding_dim)
            self.total_embed_dim += embedding_dim
        self.embeddings = nn.ModuleDict(embeddings)

        self.num_feat = num_feat

        self.task = task

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_feat - len(self.cat_cols) + self.total_embed_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10,1)
        )        

    # make predictions
    def forward(self, x):
        embedded_vector = torch.unsqueeze(x[:,:,0], -1)
        for col in range(1, self.num_feat):
            if col not in self.cat_cols:
                embedded_vector = torch.cat((embedded_vector, torch.unsqueeze(x[:,:,col], -1)), dim=2)
            else:
                # we know that this col is categorical; map each row to its embedding and concat
                embeddings = self.embeddings[str(col)](x[:,:,col].type(torch.int64))
                embedded_vector = torch.cat((embedded_vector, embeddings), dim=2)
        out = torch.squeeze(torch.sigmoid(self.linear_relu_stack(embedded_vector)), -1)
        return out

def helper(): 
    return defaultdict(dict)

# optimization code
def find_worst_case_distribution_exact(z_unsorted, rho=1.):
    '''
    find the worst-case distribution in O(n log n) time using the method from our paper
    '''
    n = len(z_unsorted)

    sorted_order = np.argsort(z_unsorted)
    z = z_unsorted[sorted_order]
    
    inverse_sorted_order = np.zeros(n, dtype=np.int64)
    for i in range(n):
        inverse_sorted_order[sorted_order[i]] = i

    # find the number k so that z1 = ... = zk < rest of zi
    k = 1
    for i in range(1, n):
        if np.isclose(z[i], z[i-1]):
            k += 1
        else:
            break

    # if it is feasible to put all the mass on the first k coordinates, we are done and m_opt=k
    if 0.5 * (n - k) * (float(n) / k) <= rho:
        p = np.zeros(n)
        p[:k] = 1./k
        return p[inverse_sorted_order]

    # else we'll continue, and we will have m > k

    m = np.arange(1, n+1)
    alpha = 2*rho*(m/n)/n + m/n - 1
    alpha_pos = (alpha > 0)

    sum_m = np.cumsum(z)
    z_m = sum_m / m
    b = np.cumsum(z ** 2)
    s_m_2_times_m_2 = m * b - (sum_m ** 2)
    s_m_2_times_m_2 = np.maximum(0, s_m_2_times_m_2)

    # lower bound from chi-squared constraint
    lmb_lower_bound_1 = np.zeros(n)
    lmb_lower_bound_1[alpha_pos] = np.sqrt(s_m_2_times_m_2[alpha_pos] / alpha[alpha_pos]) / (n ** 2)

    # lower bound from p >= 0
    lmb_lower_bound_2 = np.zeros(n)
    lmb_lower_bound_2[alpha_pos] = (m[alpha_pos] * (z[alpha_pos] - z_m[alpha_pos])) / (n ** 2)

    lmb = np.maximum(lmb_lower_bound_1, lmb_lower_bound_2)

    objective = np.inf * np.ones(n)
    objective[alpha_pos] = z_m[alpha_pos] - (s_m_2_times_m_2[alpha_pos] / m[alpha_pos]) / (lmb[alpha_pos] * (n ** 2))

    m_opt = np.argmin(objective)

    theta = (1 - float(n) / m[m_opt]) * lmb[m_opt] * n - z_m[m_opt]
    p = np.maximum(1 - (z + theta) / (lmb[m_opt] * n), 0) / n

    return p[inverse_sorted_order]


def project_onto_chi_square_ball_exact(w_unsorted, rho):
    n = len(w_unsorted)
    # if w_unsorted is already feasible, return immediately
    if 0.5 * np.sum(np.power(n * w_unsorted - 1, 2)) <= rho and np.isclose(np.sum(w_unsorted), 1) and np.all(w_unsorted >= 0):
        return w_unsorted
    # need to do the same check (as in the linear case) to see if we can put all mass on one coordinate
    sorted_order = np.argsort(-w_unsorted)
    w = w_unsorted[sorted_order]
    inverse_sorted_order = np.zeros(n, dtype=np.int64)
    for i in range(n):
        inverse_sorted_order[sorted_order[i]] = i
    w = w.astype('float64') # try to get better precision for when w is small
    m = np.arange(1, n+1)
    sum_m = np.cumsum(w)
    w_m = sum_m / m
    b = np.cumsum(w ** 2)
    s_m_2_times_m_2 = m * b - (sum_m ** 2)
    s_m_2_times_m_2 = np.maximum(0, s_m_2_times_m_2)
    alpha = (2*rho*(m/n) + m)/n - 1
    beta_sq = alpha / s_m_2_times_m_2
    valid_beta = (alpha >= 0) & (s_m_2_times_m_2 != 0)
    # valid_beta = np.array(beta_sq >= 0)
    beta = np.sqrt(beta_sq)
    beta_upper_bound_from_nonneg = -1./m / (w - w_m)
    beta_upper_bound_from_nonneg[beta_upper_bound_from_nonneg <= 0] = np.inf
    beta = np.minimum(beta, beta_upper_bound_from_nonneg)
    obj = np.inf * np.ones(n)
    obj[valid_beta] = -beta[valid_beta] * s_m_2_times_m_2[valid_beta] / m[valid_beta] - w_m[valid_beta]
    m_opt = np.argmin(obj)
    p = beta[m_opt] * (w - w_m[m_opt]) + 1./m[m_opt]
    p = np.maximum(p, 0)
    # just to help with numerics
    # p = p / np.sum(p)
    p = p.astype('float64')
    res = p[inverse_sorted_order]
    return res

def projection_simplex_sort_2(w_prime):
    e_x = np.exp(w_prime - np.max(w_prime))
    return e_x / e_x.sum()

def chi_square_divergence_exact(p):
    return 0.5 * np.sum(np.square(len(p) * p - 1))

# TODO integrate into the rest of the code
def dp_vectorized_knapsack(costs, rewards, capacity, actual_rewards):
    dp = np.zeros((capacity + 1, len(costs) + 1))
    rb = np.copy(dp)

    for i in range(len(costs)):
        dp[: costs[i], i + 1] = dp[: costs[i], i]
        rb[: costs[i],i + 1] = rb[: costs[i], i]
        prev_value = dp[: -costs[i], i] + rewards[i]
        dp[costs[i] :, i + 1] = np.maximum(dp[costs[i] :, i], prev_value)
        to_add_idxs = np.argmax(np.vstack([dp[costs[i] :, i], prev_value]), axis=0)
        rb_prev_value = rb[: -costs[i], i] + actual_rewards[i]
        helper_arr = np.vstack([rb[costs[i] :, i], rb_prev_value])
        rb[costs[i] :, i + 1] = helper_arr[to_add_idxs,np.arange(len(to_add_idxs))]
        
    return rb[np.unravel_index(dp.argmax(), dp.shape)]

def optimal_knapsack(costs, limit):
    vals = np.sort(costs)
    s = 0
    for i in range(len(vals)):
        if s > limit: return i
        s += vals[i]
    return len(vals)

def loss_knapsack(setting, df, predmodel):
    '''
    takes in the sampled data along with the state idx from which it originated, list of indexes aka individuals within the state
    returns regret: best possible reward - realized award
    '''
    # probs, education, age, label
    probs, education, age, y = df[:,0], df[:,1], df[:,2], df[:,3]

    # first isolate costs (education)
    costs = education.astype(np.int32) + 20
    total_dead = np.count_nonzero(y == 1)
    if total_dead == 0: return 0 # no regret if nobody to treat

    # solve the problem
    realized_benefit = dp_vectorized_knapsack(costs, probs, len(df) * 10, y)
    optimal_benefit = optimal_knapsack(costs[np.where(y == 1)], len(df) * 10)

    regret = 1 - (realized_benefit / optimal_benefit)
    assert(regret >= 0)
    return regret

def loss(df, predmodel):
    y = df[:,-1]
    probs = df[:,0]
    total_dead = np.count_nonzero(y == 1)

    total_selected = len(probs) // 3
    sorted_res = list(enumerate(probs))
    sorted_res.sort(key = lambda x : x[1], reverse=True)
    sorted_res = [x[0] for x in sorted_res[:total_selected]]
    true_positives = np.sum(y[sorted_res])

    modified_precision_top = true_positives / total_selected # out of all those chosen in top x pct, how many actually died?
    modified_recall_top = true_positives / total_dead if total_dead != 0 else 1 # out of all those that died, how many were selected?
    ideal_precision = min(total_dead / total_selected, 1.)
    regret = ideal_precision - modified_precision_top
    return regret

def loss_acc(df, predmodel):
    y = df[:,-1]
    probs = df[:,0]
    preds = np.round(probs)
    correct = np.sum(y == preds)
    acc = correct / len(y)
    return 1-acc

def loss_fairness(df, predmodel): # TODO verify that this works
    # treat people normally, then for each unique racial group, calculate regret and calculate a weighted sum favoring higher-regret treated races
    # if flag: pdb.set_trace()
    y = df[:,-1]
    probs = df[:,0]
    races = df[:,-2]
    unique_races = np.unique(races)

    if len(unique_races) == 1 or np.sum(y) == 0: return 0 # no opportunity for opportunity differences
    
    # modify the top-k problem
    total_selected = len(probs) // 3
    sorted_res = list(enumerate(probs))
    sorted_res.sort(key = lambda x : x[1], reverse=True)
    sorted_res = [x[0] for x in sorted_res[:total_selected]]
    treatment_vector = np.zeros(len(probs))
    treatment_vector[sorted_res] += 1 # assign treatment to those we've isolated
    new_df = np.hstack([df, np.expand_dims(treatment_vector, -1)])

    # given these labels, isolate the subsets of each race, find max possible EOD
    regrets = []
    # highest_prob = -float('inf')
    # lowest_prob = float('inf')
    for race in unique_races:
        racial_subset = new_df[new_df[:,-3] == race]
        treated_idxs = np.where(racial_subset[:,-1] == 1)
        true_positives = np.sum(racial_subset[:,-2][treated_idxs])
        total_dead = np.count_nonzero(racial_subset[:,-2] == 1)
        assert(true_positives <= total_dead)
        if total_dead == 0: continue # disregard if there was nobody to treat
        tp_prob = true_positives / total_dead
        regrets.append(tp_prob)
        # highest_prob = max(highest_prob, tp_prob)
        # lowest_prob = min(lowest_prob, tp_prob)
    # try:
    #     assert(highest_prob != -float('inf') and lowest_prob != float('inf'))
    # except:
    #     pdb.set_trace()
    # if flag: pdb.set_trace()
    # return highest_prob - lowest_prob
    # return max(regrets) - min(regrets)
    # calculate gini index
    # regrets.sort()

    # Mean absolute difference
    if np.mean(regrets) == 0: return 0 # if no regrets, perfectly equal
    mad = np.abs(np.subtract.outer(regrets, regrets)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(regrets)
    # Gini coefficient
    gini = 0.5 * rmad
    return gini

def loss_cross_entropy(df, predmodel):
    # returns average (-1/ce(df)), where ce(df) is cross-entropy loss over the sample
    high_thres = 9.999e-1
    low_thres = 1. - high_thres
    df = np.copy(df)

    # nothing outside reasonable bounds (avoiding perfectly bad/good predictions, which yield na's)
    df[:,0] = np.maximum(df[:,0], low_thres)
    df[:,0] = np.minimum(df[:,0], high_thres)

    rows = np.where(df[:,-1] == 0)
    df[rows,0] *= -1
    df[rows,0] += 1
    res = np.mean(1/np.log2(df[:,0]))
    if res > 0:
        print(f"bad ce {res}")
    return res / 100

def parallelize_state_model_pair(i):
    state, setting, test_data, numdata, optimizer, tasks, sample_size, limit, model, model_path, num_samples = i

    weights_history = defaultdict(list)
    final_weights, all_losses = {}, {}

    # 3) loop: sample from the distribution, calculate loss, and update weights
    epochs = 15 # loops to converge # TODO increase
    lr_subpop = .0000001 # learning rate for subpop specific optimization
    exp = 1 # defines exponent to which length is raised
    num_samples = num_samples # number samples to take + average over to compute loss in each epoch for a given set of weights TODO increase
    subset = test_data[state]
    assert(len(subset) == numdata)
    n = len(subset)
    rho = n # for an 80% accurate model, roughly 50% constraint
    temp = np.array([subset])
    X = torch.tensor(temp[:,:,:-1]).type(torch.float32)
    y_pred_proba = model(X).detach()[0] # sample from these, not raw data
    prev_grads = np.array([0. for _ in range(len(subset))]) # for momentum

    def set_seed(trial):
        random.seed((trial+15)*10)
        np.random.seed((trial+15)*10)

    # for each task, optimize
    for trial, task in enumerate(tasks):

        # randomization + reproducability
        set_seed(trial)
        xweights = np.array([0. for _ in range(len(test_data[state]))])
        iweights = np.array([1./len(test_data[state]) for _ in range(len(test_data[state]))])

        losses = []
    
        if task=="skim":
            loss_fn = loss
        if task=="knapsack":
            # optmodel = myModel2([math.ceil(4*(i+1)/sample_size) + (sample_size) for i in range(sample_size)], limit)
            loss_fn = partial(loss_knapsack, setting)
        if task=="acc":
            loss_fn = loss_acc
        if task=="fair":
            loss_fn = loss_fairness
        if task=="ce":
            loss_fn = loss_cross_entropy

        for epoch in tqdm(range(epochs + 1)):
            agg_loss = 0
            grads = np.array([0. for _ in range(n)])
        
            to_use_weights = xweights + iweights

            total_neg = -np.sum(to_use_weights[to_use_weights < 0])
            max_iter = 100
            indx = 0
            while total_neg > 0. and indx < max_iter:
                to_use_weights[to_use_weights < 0.] = 0.
                to_use_weights[to_use_weights > 0.] -= total_neg / (len(to_use_weights[to_use_weights > 0.]))
                total_neg = -np.sum(to_use_weights[to_use_weights < 0.])    
                indx += 1

            to_use_weights /= np.sum(to_use_weights)

            for sample_num in range(num_samples):
                selected_idxs = np.random.choice(np.array([i for i in range(len(subset))]), size=sample_size if task not in ['acc', 'ce'] else 1, p=to_use_weights, replace=True)
                sampling_freqs = Counter(selected_idxs)
                selected_sample = np.vstack([y_pred_proba[selected_idxs], subset[selected_idxs][:,(1 if setting == "employment" else 2)], subset[selected_idxs][:,-2], subset[selected_idxs][:,-1]]).T
                l_i = loss_fn(selected_sample, model) - 1 if optimizer == "fw" else loss_fn(selected_sample, model)
                if task == "ce" and optimizer == "fw": l_i += 1
                agg_loss += (l_i / num_samples)
                for idx in sampling_freqs:
                    m = sampling_freqs[idx]
                    dldwi = m*l_i/(xweights[idx] + iweights[idx]) # if gd, xweight fixed to 0; if fw, iweights fixed to 1/n
                    grads[idx] += dldwi/num_samples

            # optimize; if gradient descent, backprop and project onto feasible set; otherwise frank-wolfe
            if epoch != epochs:
                if optimizer == "gd":
                    # backprop
                    for idx, val in enumerate(grads):
                        iweights[idx] += (lr_subpop/len(iweights)) * val
                    if trial % 2 == 0:
                        proj = project_onto_chi_square_ball_exact(iweights, (len(iweights))**(exp))
                        proj /= np.sum(proj)
                    iweights = proj if trial % 2 == 0 else projection_simplex_sort_2(iweights)
                else:
                    p_t = .7
                    # frank wolfe goes here
                    neg_grad = -grads
                    altered_neg_grad = p_t*(neg_grad) + (1.-p_t)*prev_grads
                    v = find_worst_case_distribution_exact(altered_neg_grad, rho = rho)
                    v -= 1/n
                    xweights += (1./epochs) * v
            prev_grads = altered_neg_grad

            # check new weights in feasible set
            new_weights = xweights + iweights
            weights_history[task].append(new_weights)
            losses.append(agg_loss)

        to_use_weights = xweights + iweights

        total_neg = -np.sum(to_use_weights[to_use_weights < 0])
        max_iter = 100
        indx = 0
        while total_neg > 0. and indx < max_iter:
            to_use_weights[to_use_weights < 0.] = 0.
            to_use_weights[to_use_weights > 0.] -= total_neg / (len(to_use_weights[to_use_weights > 0.]))
            total_neg = -np.sum(to_use_weights[to_use_weights < 0.])    
            indx += 1

        to_use_weights /= np.sum(to_use_weights)
        final_weights[task] = to_use_weights
        all_losses[task] = losses
        # final_data[task] = [losses, to_use_weights]
    print("{} {} done".format(model_path, state))
    return final_weights, all_losses, weights_history, model_path, state

def high_level_parallel(weights_over_models, final_losses_over_models, test_data, states, optimizer, setting, sample_size, num_samples, numdata, tasks, i):
    model, model_path = i
    n2 = len(states)
    task_name_order = tasks
    relevant_weights = weights_over_models[model_path] # get access to each of the pmfs for each subpopulation, for the model
    all_sweights = {} # contains, for each loss function, converged distribution over subpopulations
    output_folder = "{}_optimized_{}_{}_{}".format(model_path, num_samples, sample_size, numdata)

    for task in relevant_weights:

        # optimize over subpopulation-level weights, holding individuals constant
        
        # concretely, use staib et al subroutine gradmax
        p=.2
        rho2 = .5*(n2*(1-p))*(1/((1-p)**2) - 2/(1-p) +1) + .5*n2*p
        final_losses = np.array([final_losses_over_models[model_path][task][state] for state in states])
        neg_losses = -final_losses
        final_sweights = find_worst_case_distribution_exact(neg_losses, rho2)
        all_sweights[task] = final_sweights
        print(list(zip(final_losses, final_sweights)))

    print("high level done with {}".format(model_path))
    # load in weights for experiment
    if 'knapsack' in all_sweights:
        knapsack_data = all_sweights['knapsack']
    if 'skim' in all_sweights:
        skim_data = all_sweights['skim']
    if 'acc' in all_sweights:
        acc_data = all_sweights['acc']
    if 'fair' in all_sweights:
        fair_data = all_sweights['fair']
    if 'ce' in all_sweights:
        ce_data = all_sweights['ce']
    

    def sample(task, loss_fn, n=300, n2=40): # TODO increase 300, 40
        '''
        thesis: the optimal distribution wrt one metric is not the same as the optimal distribution wrt another metric
        '''
        # set weights to sample from
        if task == "skim":
            weights = skim_data
        if task == "knapsack":
            weights = knapsack_data
        if task == "acc":
            weights = acc_data
        if task == "ce":
            weights = ce_data
        if task == "fair":
            weights = fair_data
        task_weights = relevant_weights[task]

        # sample n subpopulations
        subpopulations = random.choices(population=[i for i in range(len(states))], weights=weights, k=n)
        subpopulation_names = [states[i] for i in subpopulations]

        losses = []
        # for each subpopulation, calculate expected value loss given loss_fn
        for subpopulation, state in (zip(subpopulations, subpopulation_names)):
            agg_loss = 0
            # get n2 samples from the subpopulation; calculate loss on each
            subset = test_data[state]
            temp = np.array([subset])
            X = torch.tensor(temp[:,:,:-1]).type(torch.float32)
            y_pred_proba = model(X).detach()[0]
            for sample in range(n2):
                selected_idxs = np.random.choice(np.array([i for i in range(len(subset))]), size=sample_size, p=task_weights[state], replace=True)
                selected_sample = np.vstack([y_pred_proba[selected_idxs], subset[selected_idxs][:,(1 if setting == "employment" else 2)], subset[selected_idxs][:,-2], subset[selected_idxs][:,-1]]).T
                l = loss_fn(selected_sample, model)
                # if loss_fn.__name__ == "loss_cross_entropy": pdb.set_trace()
                agg_loss += l / n2
            losses.append(agg_loss)

        # return the losses and chosne subpopulations
        return losses
    
    def sample_comparison():
        task_names, loss_fns = [], []
        for task in task_name_order:
            # add in the task names, loss functions
            task_names.append(task)
            if task == "acc":
                loss_fns.append(loss_acc)
            if task == "skim":
                loss_fns.append(loss)
            if task == "knapsack":
                loss_fns.append(partial(loss_knapsack, setting))
            if task == "ce":
                loss_fns.append(loss_cross_entropy)
            if task == "fair":
                loss_fns.append(loss_fairness)
        assert(len(task_names) == len(loss_fns))
        res = [[0 for _ in range(len(task_names))] for _ in range(len(task_names))]
        for i, task in tqdm(enumerate(task_names)):
            for j, loss_fn in enumerate(loss_fns): 
                if i == j: # if main diagonal, use cached values from the fw loop
                    # set weights to sample from
                    if task == "skim":
                        weights = skim_data
                    if task == "knapsack":
                        weights = knapsack_data
                    if task == "acc":
                        weights = acc_data
                    if task == "fair":
                        weights = fair_data
                    if task == "ce":
                        weights = ce_data
                    l_i_temp = 0.
                    for k, state in enumerate(states):
                        l_i_temp += (final_losses_over_models[model_path][task][state] + 1) * weights[k]
                    res[i][j] = l_i_temp
                else:
                    l_i_temp = np.mean(sample(task, loss_fn))
                    if j == len(loss_fns) - 1: l_i_temp += 1
                    res[i][j] = l_i_temp
                
        return np.array(res)
    
    final_vals = sample_comparison()
    print(final_vals) # TODO delete
    with open('{}/final_results.pickle'.format(output_folder), 'wb') as handle:
        pickle.dump(final_vals, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.matshow(final_vals)
    cb = plt.colorbar()
    plt.title("Aggregate Results for Setting {}".format(model_path))
    loss_names = task_name_order
    ticks = [i for i in range(len(loss_names))]
    plt.xticks(ticks, loss_names, rotation=20)
    plt.yticks(ticks, loss_names, rotation=20)
    plt.xlabel("Loss Function")
    plt.ylabel("Distribution")
    plt.savefig("{}/final_results_vis.pdf".format(output_folder), format="pdf", bbox_inches='tight')
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')   
    gc.collect()
    print("finished high-level on model {}".format(model_path))
    return all_sweights, model_path # TODO figure out how to integrate this into a test_converged_weights_{task}.pickle file

if __name__ == "__main__":
    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"][:5]

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_size")
    parser.add_argument("num_samples")
    parser.add_argument("modelpath")
    parser.add_argument("optimizer")
    parser.add_argument("task")
    parser.add_argument("numdata")
    parser.add_argument("setting")
    args = parser.parse_args()
    sample_size = int(args.sample_size)
    num_samples = int(args.num_samples)
    model_paths = args.modelpath.split(".")
    optimizer = args.optimizer
    setting = args.setting
    tasks = args.task
    numdata = int(args.numdata)
    tasks = tasks.split(".")

    if optimizer not in ["fw", 'gd']:
        raise Exception("not a valid optimizer, please use gradient descent (gd) or frank-wolfe (fw)")
    
    for task in tasks:
        if task not in ["knapsack", "skim", 'acc', 'fair', 'ce']:
            raise Exception("not a valid decision task")
        
    # load in data
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    test_data = {}
    individual_weights = {}

    for i, state in tqdm(list(enumerate(states))):
        acs_data = data_source.get_data(states=[state], download=False)
        features, label, group = ACSEmployment.df_to_numpy(acs_data) if setting == "employment" else ACSIncome.df_to_numpy(acs_data)

        X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
            features, label, group, test_size=0.2, random_state=0)
        if setting == "income": X_train, X_test = np.delete(X_train, [4,5], axis=1), np.delete(X_test, [4,5], axis=1) # removing cols occupation bc way too complicated to encode each occupation; removing each state of birth also bc way too much to encode

        X_test, y_test = X_test[:numdata], y_test[:numdata]

        test_data[state] = np.hstack((X_test, (1. - y_test).reshape(-1,1)))

    one_cols = [2,4,6,9,10,11,12,14,15] if setting == "employment" else [1,3,6,7]
    for state in test_data:
        for col in one_cols:
            test_data[state][:,col] -= 1
    
    # for each model, get the necessary inputs and collect into a list to pass into the parallelized function
    '''
    test_data
    individual_weights
    numdata
    optimizer
    task
    sample_size
    limit
    model_path
    '''

    # each element should represent optimizing all loss functions over a single (model, state) pair.
    # we build these pairs and optimize them (#vCPUs) at a time
    args = []
    for model_path in tqdm(model_paths):
        #experiment params
        model = torch.load("{}/{}.pt".format(model_path, model_path))
        optimizer = optimizer
        # sample_size = 20
        limit = 10*sample_size
        for state in states:
            args.append([state, setting, test_data.copy(), numdata, optimizer, tasks, sample_size, limit, model, model_path, num_samples])

        #filename checking
        output_folder = "{}_optimized_{}_{}_{}".format(model_path, num_samples, sample_size, numdata)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for task in tasks:
            if not os.path.exists("{}/{}".format(output_folder, task)):
                os.makedirs("{}/{}".format(output_folder, task))

    weights_over_models = defaultdict(helper) # contains, for each model, a dict over tasks going to a dict of states of converged weights (argmax solution)
    weights_history_over_models = defaultdict(helper) # contains, for each model/task/state, list of prior weights 
    final_losses_over_models = defaultdict(helper) # contains, for each model/task/state, the converged loss (max solution)

    # optimize over models in parallel (you will need to perform the post-hoc analysis here as well)
    for i in tqdm(range(0, len(args), 64)):
        batch_args = args[i:i+64]
        assert(len(batch_args) <= 64)
        results = []
        # for arg in batch_args:
        #     results.append(parallelize_state_model_pair(arg))
        with ProcessPoolExecutor() as pool:
            results = pool.map(parallelize_state_model_pair, batch_args)
        print("finished batch {}".format(i // 64))
        
        start = time.perf_counter()
        for final_weights, all_losses, weights_history, model_path, state in results:
            output_folder = "{}_optimized_{}_{}_{}".format(model_path, num_samples, sample_size, numdata)
            for task in tasks:
                # for each model-state pair, plot performance on the state
                losses = all_losses[task]
                plt.plot([i for i in range(len(losses))], losses)
                plt.savefig("{}/{}/{}.pdf".format(output_folder, task, state), format="pdf", bbox_inches='tight')
                # Clear the current axes.
                plt.cla() 
                # Clear the current figure.
                plt.clf() 
                # Closes all the figure windows.
                plt.close('all')   
                gc.collect()                
                # write back prior weights, final weights, last loss
                converged_weights = final_weights[task]
                weights_incurred = weights_history[task]
                weights_over_models[model_path][task][state] = converged_weights
                weights_history_over_models[model_path][task][state] = weights_incurred
                final_losses_over_models[model_path][task][state] = losses[-1]
        end = time.perf_counter()
        print("{} s write".format(end - start))

    # for each optimized model, perform high-level optimization, then evaluation, in parallel
    print("optimizing models")
    
    optim_start = time.perf_counter()
    high_level_args = []
    for model_path in model_paths:
        model = torch.load("{}/{}.pt".format(model_path, model_path))
        high_level_args.append([model, model_path])
    print(len(high_level_args))
    results = []
    with ProcessPoolExecutor() as pool: # parallelize high-level optimization over models
        results = pool.map(partial(high_level_parallel, weights_over_models, final_losses_over_models, test_data, states, optimizer, setting, sample_size, num_samples, numdata, tasks), high_level_args)
    for all_sweights, model_path in results:
        output_folder = "{}_optimized_{}_{}_{}".format(model_path, num_samples, sample_size, numdata)
        for task in all_sweights:
            weights_over_models[model_path][task]['all'] = all_sweights[task]
            # save the model's full set of weights to pkl
            with open('{}/test_converged_weights_{}.pickle'.format(output_folder, task), 'wb') as handle:
                pickle.dump(weights_over_models[model_path][task], handle, protocol=pickle.HIGHEST_PROTOCOL)
    optim_end = time.perf_counter()
    print("optimized {} models in {} min".format(len(high_level_args), (optim_end - optim_start)/60))
        
