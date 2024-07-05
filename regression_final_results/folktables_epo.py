from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from folktables2.folktables import ACSDataSource, ACSEmployment, ACSIncome
from sklearn.model_selection import train_test_split
import warnings
import argparse
import os
import math
import json
import time
from tqdm import tqdm
from torch import nn
from folktables_abcmodule import optModule
from folktables_spoplus import SPOPlus
from folktables_knapsack_dataset import optDataset

from multiprocessing import Pool
from functools import partial
# from utils import LogisticRegressionEmployment, NeuralNetworkIncome
warnings.filterwarnings('ignore')

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
        # return torch.maximum(torch.tensor(0), 70000. - out)

def sampler(sub, rewards, sample_size, task, i):
    selected_idxs = np.random.choice(np.array([i for i in range(len(sub))]), size=sample_size, replace=True)
    selected_sample = np.array([sub[selected_idxs]]) # (1,20,16)
    # selected_rewards = np.maximum(0, 70000. - rewards[selected_idxs]) # (20,)
    selected_rewards = rewards[selected_idxs]
    selected_costs = selected_sample[0,:,(1 if task == 'employment' else 2)] + 20 # (20,)

    # # join the arrays, sort by education, and split again
    # combined = np.hstack((selected_sample[0], selected_rewards)) # (20, 17)
    # sorted_combined = combined[(-combined[:, (1 if task == 'employment' else 2)]).argsort()] # (20, 17)
    # selected_sample = np.expand_dims(sorted_combined[:,:-1], 0) # (1, 20, 16)
    # selected_rewards = np.expand_dims(sorted_combined[:,-1], -1) # (20, 1)
    return selected_sample, selected_rewards, selected_costs

if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("filename")
    parser.add_argument("state")
    args = parser.parse_args()
    task = args.task
    model_state = args.state
    filename = args.filename

    # create output folder if it doesn't exist
    if not os.path.exists(filename):
        os.makedirs(filename)
    max_min = {2: 5, 3: 18, 4: 2, 5: 9, 6: 5, 7: 4, 8: 5, 9: 8, 10: 2, 11: 2, 12: 2, 13: 3, 14: 2, 15: 9} if task == "employment" else {1: 8, 3: 5, 4: 18, 6: 2, 7: 9}
    
    # loading in data
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    state_data = {} # map state abbev to (X_train, X_test, y_train, y_test, group_train, group_test)

    train_data = {}
    test_data = {}
    individual_weights = {}
    rewards = {} # contains reward for each individual if treated
    
    # for each state, read in data
    acs_data = data_source.get_data(states=[model_state], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data) if task == "employment" else ACSIncome.df_to_numpy(acs_data)
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        features, label, group, test_size=0.2, random_state=0)
    if task == "income": X_train, X_test = np.delete(X_train, [4,5], axis=1), np.delete(X_test, [4,5], axis=1) # removing cols occupation bc way too complicated to encode each occupation; removing each state of birth also bc way too much to encode
    X_al, y_al = X_train, 0. - y_train
    one_cols = [2,4,6,9,10,11,12,14,15] if task == "employment" else [1,3,6,7]
    for col in one_cols:
        X_al[:,col] -= 1
        assert(np.amin(X_al[:,col]) == 0)
    
    # generate predict then optimize dataset from training data
    num_data = 15000 # number of data/samples we take
    num_feat = 8 if task == "income" else 16 # size of feature
    num_item = 40 # number of items/individuals per sample
    limit = 10*num_item # cost we're willing to incur per instance

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

    def loss_knapsack(costs, probs, y):
        '''
        takes in the sampled data along with the state idx from which it originated, list of indexes aka individuals within the state
        returns regret: best possible reward - realized award
        '''
        probs = np.maximum(0, 70000 - probs) # motivate maximizing sum of probs
        y = np.maximum(0, 70000 - probs)

        realized_benefit = dp_vectorized_knapsack(costs, probs, len(costs) * 10, y)
        optimal_benefit = dp_vectorized_knapsack(costs, y, len(costs) * 10, y) # in this, benefits are known == y
        regret = 1 - (realized_benefit / optimal_benefit)
        return regret

    def optimal_knapsack_solver(costs, weights, limit=200):
        # costs: outcome variable, weights: incurred cost of treatment
        # takes in the capacity, labels, weights associated with a set of individuals.
        # return the optimal allocation along with max number of individuals that can be treated.
        # limit: int, costs: (20,), weights: (20,)
        costs = np.maximum(0, 70000. - costs)

        def set_val(t, n, w, val, line):
            # print(f"Setting cache {n} {w} value: {val}: line num {line}")
            t[n][w] = val

        def helperSolve(wt, val, W, n):
            t = [[-1 for i in range(W + 1)] for j in range(n + 1)] 
            def knapsack(wt, val, W, n, path=set()): 

                # base conditions 
                if n == 0 or W == 0:
                    return 0, set()
                if t[n][W] != -1:
                    return t[n][W] 
            
                # choice diagram code 
                if wt[n-1] <= W: 
                    notUsePath = path.copy()
                    notUseObj, notUsePath = knapsack(wt, val, W, n-1, set())
                    notUsePath = notUsePath.copy()
                    useObj, usePath = knapsack(wt, val, W-wt[n-1], n-1, set())
                    usePath = usePath.copy()
                    usePath.add(n)
                    useObj += val[n-1]
                    
                    if useObj > notUseObj:
                        set_val(t, n, W, (useObj, usePath), 35)
                    else:
                        set_val(t, n, W, (notUseObj, notUsePath), 38)
                    return t[n][W] 
                elif wt[n-1] > W: 
                    set_val(t, n, W, knapsack(wt, val, W, n-1, set()), 42)
                    return t[n][W] 
            obj, sol = knapsack(wt, val, W, n)
            res_sol = np.zeros(n)
            res_sol[[int(x) - 1 for x in sol]] = 1
            return res_sol, obj
        
        return helperSolve(weights, costs, limit, len(costs))
    
    # init prediction model
    predmodel = NeuralNetworkIncome(max_min, num_feat, 'epo')
    for key in predmodel.embeddings:
        embeds = predmodel.embeddings[key]
        embeds.weight.data.zero_() # NOTE hopefully zero init is better than random init????
    
    start = time.perf_counter()
    with Pool() as pool:
        x = pool.map(partial(sampler, X_al, y_al, num_item, task), map(np.random.default_rng, range(num_data)))
    xs = [item[0] for item in x]
    c = [item[1] for item in x]
    weights = [item[2] for item in x]
    end = time.perf_counter()

    x, c = np.squeeze(np.array(xs), 1), np.array(c).astype(np.float64)

    # set optimizer
    optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-3)
    # build dataset
    dataset = optDataset(x, c, weights, optimal_knapsack_solver)
    # init SPO+ loss
    spop = SPOPlus(dataset=dataset) # no using the optmodel TODO initialize with custom dataset
    # get data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    losses = []
    regrets = []

    # write metadata to json
    json_obj = {
        "num_data": num_data, # number of data/samples we take
        "num_feat": num_feat,# size of feature
        "num_item": num_item, # number of items/individuals per sample
        "limit": limit, # cost we're willing to incur per instance
        "task": task
    }
    with open("{}/metadata.json".format(filename), 'w') as f:
        json.dump(json_obj, f)

    # training
    num_epochs = 5
    for epoch in tqdm(range(num_epochs)):
        for data in dataloader:
            x, c, w, z, weights = data # features, labels, optimal allocations, optimal objs, weights
            # forward pass
            cp = predmodel(x)
            loss = spop(cp, c, w, z, weights)
            losses.append(float(loss))
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # regret calculation
        predmodel.eval()
        agg_regret = 0
        for data in dataloader:
            x, c, w, z, weights = data
            first_entry = torch.tensor(np.expand_dims(x[0],0))
            y_pred_proba = predmodel(first_entry)
            regret = loss_knapsack(weights[0].int().detach().numpy(), y_pred_proba[0].detach().numpy(), c[0].detach().numpy())
            agg_regret += regret / len(dataloader)
            # find the optimal/achieved benefits; calculate regret


            # # set objective function in model
            # optmodel.setObj(y_pred_proba[0])
            # sol, _ = optmodel.solve()

            # # solve for allocation, realized benefit
            # realized_benefit = np.dot(sol, c[0])

            # # find optimal benefit, calculate regret and return
            # optmodel.setObj(c[0])
            # optSol, optimal_benefit = optmodel.solve()
            # regret = 1 - (realized_benefit / optimal_benefit)
            # agg_regret += regret / len(dataloader)
        regrets.append(agg_regret)
        predmodel.train()

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            torch.save(predmodel, "{}/{}.pt".format(filename, filename))

        plt.plot([i for i in range(len(losses))], losses)
        plt.title("Training Loss: basic knapsack predict-then-optimize")
        plt.savefig("{}/spo_loss_curve.png".format(filename))
        plt.close('all')

        plt.plot([i for i in range(len(regrets))], regrets)
        plt.title("Training Regret: basic knapsack predict-then-optimize")
        plt.savefig("{}/spo_regret_curve.png".format(filename))
        plt.close('all')
    print("saved to {}".format(filename))
