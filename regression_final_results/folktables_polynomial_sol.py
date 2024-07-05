import gc
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
SolverFactory('ipopt').available(exception_flag=True)
import subprocess
import pickle
import numpy as np
import torch
import torch.nn as nn
from folktables2.folktables import ACSDataSource, ACSEmployment, ACSIncome
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from functools import partial
import math
import os
from matplotlib import pyplot as plt
import time
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

plt.rc("axes",titlesize=20)
plt.rc("axes",labelsize=20)
plt.rc("font",size=14)

start_time = time.perf_counter()

class NeuralNetworkIncome(nn.Module):    
    # build the constructor
    def __init__(self, max_min, num_feat, task):
        super().__init__()

        # create embeddings
        self.cat_cols = [1,3,4,6,7]
        self.embedding_size_dict = max_min
        self.embedding_dim_dict= {key: 20 for key,val in self.embedding_size_dict.items()}

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
            nn.Linear(num_feat - len(self.cat_cols) + self.total_embed_dim, 35),
            nn.ReLU(),
            nn.Linear(35, 20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )        

    # make predictions
    def forward(self, x):
        embedded_vector = torch.unsqueeze(x[:,:,0], -1)
        for col in range(1, self.num_feat): # TODO make not hacky (range hardcoded for employment)
            if col not in self.cat_cols:
                embedded_vector = torch.cat((embedded_vector, torch.unsqueeze(x[:,:,col], -1)), dim=2)
            else:
                # we know that this col is categorical; map each row to its embedding and concat
                embeddings = self.embeddings[str(col)](x[:,:,col].type(torch.int64))
                embedded_vector = torch.cat((embedded_vector, embeddings), dim=2)
        out = torch.squeeze(self.linear_relu_stack(embedded_vector), -1)
        return out

'''LOSS FUNCTIONS'''
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

def dp_vectorization(weights, values, capacity):
    """Dynamic programming algorithm with numpy array and vectorization"""
    dp = np.zeros((capacity + 1, len(weights) + 1), dtype=np.uint32)

    for i in range(len(weights)):
        dp[: weights[i], i + 1] = dp[: weights[i], i]
        prev_value = dp[: -weights[i], i] + values[i]
        dp[weights[i] :, i + 1] = np.maximum(dp[weights[i] :, i], prev_value)
    return np.amax(dp)

def loss_knapsack(setting, df, predmodel):
    '''
    takes in the sampled data along with the state idx from which it originated, list of indexes aka individuals within the state
    returns regret: best possible reward - realized award
    '''
    # probs, education, age, label
    probs, education, age, y = df[:,0], df[:,1], df[:,2], df[:,3]
    probs = np.maximum(0, 70000 - probs) # motivate maximizing sum of probs
    y = np.maximum(0, 70000 - y) # TODO validate that this is correct
    if np.sum(y) == 0: return 0 # nobody to treat -- everybody has income above thres

    # first isolate costs (education)
    costs = education.astype(np.int32) + 20

    # solve the problem
    realized_benefit = dp_vectorized_knapsack(costs, probs, len(df) * 10, y)
    # optimal_benefit = dp_vectorized_knapsack(costs, y, len(df) * 10, y) # in this, benefits are known == y
    optimal_benefit = dp_vectorization(costs, y, len(df)*10)

    regret = 1 - (realized_benefit / optimal_benefit)
    if regret < 0: # THIS SHOULD NOT HAPPEN
        print(regret)
        print(realized_benefit)
        print(optimal_benefit)
        print(probs)
    assert(regret >= 0)
    return regret

def loss(df, predmodel):
    y = df[:,-1]
    probs = df[:,0]

    total_selected = len(probs) // 3
    sorted_res = list(enumerate(probs))
    sorted_res.sort(key = lambda x : x[1]) # treat the poorest people
    sorted_res = [x[0] for x in sorted_res[:total_selected]]
    realized_sum = np.sum(y[sorted_res])

    # get actual poorest people
    sorted_label = list(enumerate(y))
    sorted_label.sort(key = lambda x : x[1])
    sorted_label = [x[0] for x in sorted_label[:total_selected]]
    ideal_sum = np.sum(y[sorted_label])

    regret = 1 - ideal_sum / realized_sum # as realized_sum gets worse (greater), loss increases

    return regret

def loss_fairness(df, predmodel): # TODO verify that this works
    # treat people normally, then for each unique racial group, calculate regret and calculate a weighted sum favoring higher-regret treated races
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

    # calculate optimal top-k
    optimal_treatment_vector = np.zeros(len(probs))
    sorted_labels = list(enumerate(y))
    sorted_labels.sort(key = lambda x: x[1], reverse=True)
    sorted_labels = [x[0] for x in sorted_labels[:total_selected]]
    optimal_treatment_vector[sorted_labels] += 1
    new_df = np.hstack([new_df, np.expand_dims(optimal_treatment_vector, -1)]) # pred, education, race, label, treatment, optimal treatment

    # given these labels, isolate the subsets of each race, find max possible EOD
    regrets = []
    # highest_prob = -float('inf')
    # lowest_prob = float('inf')
    for race in unique_races:
        racial_subset = new_df[new_df[:,-4] == race]
        treated_idxs = np.where(racial_subset[:,-2] == 1)
        true_positives = np.sum(racial_subset[:,-1][treated_idxs])
        total_dead = np.count_nonzero(racial_subset[:,-1] == 1)
        assert(true_positives <= total_dead)
        if total_dead == 0: continue # disregard if there was nobody to treat
        tp_prob = true_positives / total_dead
        regrets.append(tp_prob)

    # Mean absolute difference
    if np.mean(regrets) == 0: return 0 # if no regrets, perfectly equal
    mad = np.abs(np.subtract.outer(regrets, regrets)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(regrets)
    # Gini coefficient
    gini = 0.5 * rmad
    return gini

def loss_utility(df, predmodel):
    # water-filling method, after which we take geometric mean of money/threshold -> utility

    budget_original = 150000 # how much do I have available to allocate?
    threshold = 50000 # threshold which yields 0 utility
    budget = budget_original

    # how much utility did each household have to begin with?
    # original_utilities = np.minimum(threshold, df[:,-1]) + .2*(np.log2(np.maximum(1, df[:,-1] - threshold))**3)

    # 1) get candidate treatment
    treatments = np.zeros(len(df)).astype(float)
    sorted_by_pred = df[df[:,0].argsort()] # sort by predicted income, increasing
    for i in range(1, len(df)): # should be able to assert that, at a given index, all entries below are identical
        if budget == 0: break # nothing left to give
        first_val = treatments[0] + sorted_by_pred[0,0] # should be equal up to i exclusive
        to_give = min(sorted_by_pred[i,0] - first_val, budget / i)
        treatments[:i] += to_give
        budget -= to_give * i # subtract given amount from budget
    
    treatments += (budget / len(df)) # give out rest equally, if still money left

    # 2) realized utility under labels
    after_treatment_vals = sorted_by_pred[:,-1] + treatments
    # realized_utilities = np.minimum(threshold, after_treatment_vals) + .2*(np.log2(np.maximum(1, after_treatment_vals - threshold))**3) # diminishing returns after threshold
    # realized_utility = np.mean(realized_utilities - original_utilities[df[:,0].argsort()])
    realized_utility = np.exp(np.mean(np.log(after_treatment_vals / threshold)))

    # 3) optimal utility substituting label for pred
    budget = budget_original

    optimal_treatments = np.zeros(len(df)).astype(float)
    sorted_by_label = df[df[:,-1].argsort()]
    for i in range(1, len(df)):
        if budget == 0: break # nothing left to give
        first_val = optimal_treatments[0] + sorted_by_label[0,-1] # should be equal up to i exclusive
        to_give = min(sorted_by_label[i,-1] - first_val, budget / i)
        optimal_treatments[:i] += to_give
        budget -= to_give * i # subtract given amount from budget
    
    optimal_treatments += (budget / len(df)) # give out rest equally, if still money left  
    after_treatment_optimal = sorted_by_label[:,-1] + optimal_treatments
    # realized_utilities_optimal = np.minimum(threshold, after_treatment_optimal) + .2*(np.log2(np.maximum(1, after_treatment_optimal - threshold))**3) # diminishing returns after threshold
    # realized_utility_optimal = np.mean(realized_utilities_optimal - original_utilities[df[:,-1].argsort()])
    realized_utility_optimal = np.exp(np.mean(np.log(after_treatment_optimal / threshold)))

    # 4) regret
    regret = np.maximum(1 - (realized_utility / realized_utility_optimal), 0)
    return regret

def loss_mse(df, predmodel): # TODO how to make this bounded 0, 1? try logging the mse, then divide by like 10e9
    preds, y = df[:,0], df[:,-1]
    mse = np.mean((preds - y)**2)
    modified_mse = np.log2(mse) / 10**2
    return np.minimum(1., modified_mse)

def parallelize_pyomo(arr):
    allocations, numdata, test_data, model_name, loss_name, loss_fn_code, states = arr

    # get all allocations w/loss vals
    ls = []
    possible_allocations = {}
    for state in tqdm(states):
        possible_allocations_temp = []
        for alloc in allocations:
            freqs = list(alloc)
            idxs = []
            for i in range(len(freqs)):
                for _ in range(freqs[i]):
                    idxs.append(i)
            assert(len(idxs) == numdata)
            df = test_data[state][idxs]
            # pass in sample to get loss_fn
            l_i = loss_fn_code(df, None)
            ls.append(l_i)
            num_perm = math.factorial(numdata) / np.prod(np.array([math.factorial(x) for x in freqs]))

            # append data to all allocations
            possible_allocations_temp.append(freqs + [l_i] + [num_perm])
        possible_allocations[state] = possible_allocations_temp

    # if loss_name in final_objs[model_name].keys():
    #     return model_name, loss_name, final_objs[model_name][loss_name], None, None, possible_allocations

    # generate the pyomo model
    model = pyo.ConcreteModel()
    model.x = pyo.VarList(domain=pyo.Integers)
    for _ in range(numdata):
        model.x.add()

    # add constraints
    model.simplexbound = pyo.Constraint(expr = sum(model.x[i+1] for i in range(numdata)) == 1)
    model.posbound = pyo.ConstraintList()
    for i in range(numdata):
        model.posbound.add(expr = model.x[i+1] >= 0)
    model.chisqbound = pyo.Constraint(expr = .5*sum((numdata * model.x[i+1] - 1)**2 for i in range(numdata)) <= numdata)

    obj_vals = []

    for state in tqdm(states):
        model.obj = pyo.Objective(expr = sum((model.x[1]**i)*(model.x[2]**j)*(model.x[3]**k)*(model.x[4]**l)*(model.x[5]**m)\
                                            *(model.x[6]**n)*(model.x[7]**o)*(model.x[8]**p)*l_i*num_perm for i,j,k,l,m,n,o,p,l_i,num_perm in possible_allocations[state]), sense=pyo.maximize)

        opt = SolverFactory('ipopt', solver_io='nl')
        opt.options['max_iter'] = 2000
        opt.solve(model, tee=False) 

        res = model.obj() # store the final solved value
        obj_vals.append(res)

    # manually perform high-level optimization; calculate expected loss over instances

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
    p=.2
    rho2 = .5*(len(states)*(1-p))*(1/((1-p)**2) - 2/(1-p) +1) + .5*len(states)*p
    worst_dist = find_worst_case_distribution_exact(-np.array(obj_vals), rho=rho2)

    final_obj = np.dot(worst_dist, np.array(obj_vals))
    return model_name, loss_name, final_obj, worst_dist, obj_vals, possible_allocations

if __name__ == "__main__":

    '''HYPERPARAMS'''
    numdata = 8 # n_j
    loss_fns = ['knapsack', 'top-k', 'util', 'fair', 'mse'] # TODO logistically, how are we going to expand this to multiple loss fns?
    model_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"][:5]
    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"][:5]
    setting = "income"
    tasks = ["CE", "SPO"]
    model_names = []

    for model_state in model_states:
        for task in tasks:
            model_names.append(f"{model_state}_{task}_{setting}") # each model_name gives us the state, task, and setting, which we can regex out

    sample_nums = [10,100,300,1000,1500,2000,3000]

    '''GENERATING OUR MODELS'''

    loss_fn_str = ".".join(loss_fns)

    # for model_name in model_names:
    for sample_num in sample_nums:
        # don't run experiment if done already
        not_done_models = []
        for model_name in model_names:
            output_folder = "{}_optimized_{}_{}_{}".format(model_name, sample_num, numdata, numdata)
            if not os.path.exists(output_folder):
                not_done_models.append(model_name)
        if not not_done_models: continue

        # run experiments on unfinished runs
        model_names_str = ".".join(not_done_models) 
        cmd = f"python3 folktables_copy_parallel.py {numdata} {sample_num} {model_names_str} fw {loss_fn_str} {numdata} income"
        subprocess.run(cmd.split(" "))  

    '''LOADING IN DATA'''

    # load in data TODO for the binary prediction, actually load in the data twice (inc, emp)
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    raw_data = {}

    for i, state in tqdm(list(enumerate(states))):
        acs_data = data_source.get_data(states=[state], download=False)
        features, label, group = ACSEmployment.df_to_numpy(acs_data) if setting == "employment" else ACSIncome.df_to_numpy(acs_data)

        X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
            features, label, group, test_size=0.2, random_state=0)
        if setting == "income": X_train, X_test = np.delete(X_train, [4,5], axis=1), np.delete(X_test, [4,5], axis=1) # removing cols occupation bc way too complicated to encode each occupation; removing each state of birth also bc way too much to encode

        X_test, y_test = X_test[:numdata], y_test[:numdata]

        raw_data[state] = np.hstack((X_test, (0. - y_test).reshape(-1,1)))

    one_cols = [2,4,6,9,10,11,12,14,15] if setting == "employment" else [1,3,6,7]
    for state in raw_data:
        for col in one_cols:
            raw_data[state][:,col] -= 1

    agg_results = {model_name: {loss_fn : None for loss_fn in loss_fns} for model_name in model_names}
    final_objs = {model_name: {loss_fn: None for loss_fn in loss_fns} for model_name in model_names}

    data_over_models = {}
    for model_name in model_names:
        predmodel = torch.load("{}/{}.pt".format(model_name, model_name))

        test_data = {}

        for state in states:
            # load in the data and get the necessary attributes
            subset = raw_data[state]
            temp = np.array([subset])
            X = torch.tensor(temp[:,:,:-1]).type(torch.float32)
            y_pred_proba = predmodel(X).detach()[0]
            if "SPO" in model_name: y_pred_proba = np.maximum(0, 70000. - y_pred_proba)
            selected_sample = np.vstack([y_pred_proba, subset[:,(1 if setting == "employment" else 2)], subset[:,-2], subset[:,-1]]).T
            test_data[state] = selected_sample
        data_over_models[model_name] = test_data # model, state, np.darray

    '''GETTING ALL SOLUTIONS IN PARALLEL, WITH PYOMO'''

    # get all possible allocations, with replacement

    from itertools import permutations

    def combosWithSum(sum):
        def buildCombosWithSum(sum, combo, result):
            for num in range(sum, 0, -1):
                combo.append(num)
                if num == sum:
                    result.append(combo[:])
                else:
                    buildCombosWithSum(sum - num, combo, result)
                combo.pop()

        if sum < 0:
            raise ValueError("Sum cannot be negative: " + str(sum))
        if sum == 0:
            return []
        result = []
        buildCombosWithSum(sum, [], result)

        allSols = set()
        for sol in result:
            # preprocess w 0s
            if len(sol) < numdata:
                for _ in range(numdata - len(sol)):
                    sol.append(0)
            if tuple(sol) in allSols: continue
            # print(sol)
            # permute
            perms = permutations(sol)
            for perm in perms:
                allSols.add(perm)

        return allSols

    allocations = combosWithSum(numdata)

    # accumulate arguments
    args_compiled = []
    for model_name in model_names:
        for loss_fn in loss_fns:
            if loss_fn == "mse":
                loss_fn_code = loss_mse
            if loss_fn == "knapsack":
                loss_fn_code = partial(loss_knapsack, "income")
            if loss_fn == "util":
                loss_fn_code = loss_utility
            if loss_fn == "top-k":
                loss_fn_code = loss
            if loss_fn == "fair":
                loss_fn_code = loss_fairness

            args = [allocations, numdata, data_over_models[model_name], model_name, loss_fn, loss_fn_code, states]
            args_compiled.append(args)

    # run pyomo in parallel
    all_results = []
    for arg in tqdm(args_compiled):
        all_results.append(parallelize_pyomo(arg))

    all_allocations = {model_name: {loss_fn: None for loss_fn in loss_fns} for model_name in model_names}
    for model_name, loss_name, final_obj, worst_dist, obj_vals, possible_allocations in all_results:
        all_allocations[model_name][loss_name] = possible_allocations
        final_objs[model_name][loss_name] = final_obj

    # save pyomo results
    with open(f'final_objs_{setting}.pickle', 'wb') as handle:
        pickle.dump(final_objs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for model_name in model_names:

        for loss_fn in loss_fns:

            '''DO THE MATH, WITH OUR CODE'''

            # call the solver as a subroutine NOTE assumes that test_state is contained within 'states' in copy parallel

            converged_objs = []

            for sample_num in sample_nums:
                output_folder = "{}_optimized_{}_{}_{}".format(model_name, sample_num, numdata, numdata)

                # read in the weights 
                with open(f"{output_folder}/test_converged_weights_{loss_fn}.pickle", 'rb') as handle:
                    converged_weights = pickle.load(handle)

                # get the converged objective value
                realized_objs = []
                for state in states:
                    tmp_weights = converged_weights[state]
                    assert(len(tmp_weights) == numdata)
                    realized_sol = sum((tmp_weights[0]**i)*(tmp_weights[1]**j)*(tmp_weights[2]**k)*(tmp_weights[3]**l)*(tmp_weights[4]**m)\
                                    *(tmp_weights[5]**n)*(tmp_weights[6]**o)*(tmp_weights[7]**p)*l_i*num_perm for i,j,k,l,m,n,o,p,l_i,num_perm in all_allocations[model_name][loss_fn][state])
                    realized_objs.append(realized_sol)

                assert(len(converged_weights['all']) == len(states))
                realized_obj = np.dot(converged_weights['all'], np.array(realized_objs))

                # compare to the solved value
                converged_objs.append(realized_obj)
            agg_results[model_name][loss_fn] = converged_objs

    # save agg results
    with open(f'agg_results_{setting}.pickle', 'wb') as handle:
        pickle.dump(agg_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    '''POSTPROCESSING/PLOTTING'''
    print("plotting now")
    # compiling results
    agg_results_grid = {model_type: {loss_fn: [] for loss_fn in loss_fns} for model_type in tasks}
    final_objs_grid = {model_type: {loss_fn: [] for loss_fn in loss_fns} for model_type in tasks}

    for model_path in model_names:
        if "CE" in model_path:
            key = "CE"
        else:
            assert("SPO" in model_path)
            key = "SPO"
        for loss_fn in loss_fns:
            agg_results_grid[key][loss_fn].append(agg_results[model_path][loss_fn])
            final_objs_grid[key][loss_fn].append(final_objs[model_path][loss_fn])

    for key in tasks:
        for loss_fn in loss_fns:
            agg_results_grid[key][loss_fn] = [np.mean([x[i] for x in agg_results_grid[key][loss_fn]]) for i in range(len(sample_nums))]
            final_objs_grid[key][loss_fn] = np.mean(final_objs_grid[key][loss_fn])

    with open(f"agg_results_grid_{setting}.pickle", 'wb') as handle:
        pickle.dump(agg_results_grid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"obj_vals_grid_{setting}.pickle", 'wb') as handle:
        pickle.dump(final_objs_grid, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # plotting
    output_folder = "efficiency_exps_combined"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for task in tasks:
        for loss_fn in loss_fns:
            ests_wrt_sample_size = agg_results_grid[task][loss_fn]
            ref = final_objs_grid[task][loss_fn]
            ests_wrt_sample_size = [x / ref for x in ests_wrt_sample_size]

            plt.plot(sample_nums, ests_wrt_sample_size, label=loss_fn)
        plt.plot(sample_nums, [1 for _ in range(len(sample_nums))], label="optimal")
        plt.legend()
        plt.xlabel("Number samples per iter FW")
        plt.ylabel("Converged Loss of Model")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{task}_{setting}.pdf", format="pdf", bbox_inches="tight")
        # Clear the current axes.
        plt.cla() 
        # Clear the current figure.
        plt.clf() 
        # Closes all the figure windows.
        plt.close('all')   
        gc.collect()

    end_time = time.perf_counter()
    print("finished in {} m".format((end_time - start_time)/ 60))
