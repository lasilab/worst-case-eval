from matplotlib import pyplot as plt        
from folktables import ACSDataSource, ACSEmployment, ACSIncome
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import random
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import torch
from torch import nn
import gc
import os

# plotting adjustments
plt.rc("axes",titlesize=20)
plt.rc("axes",labelsize=20)
plt.rc("font",size=14)

# predictive models
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

if __name__ == "__main__":
    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"][:5]
    tasks = ["employment", "income"]

    for k, task in enumerate(tasks):
        # loading in data
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        state_data = {} # map state abbev to (X_train, X_test, y_train, y_test, group_train, group_test)

        train_data = {}
        test_data = {}
        individual_weights = {}
        costs = {} # contains cost to treat each individual
        rewards = {} # contains reward for each individual if treated
        finalfinalweights = {}
        numdata = 25
        # for each state, read in data
        for i, state in tqdm(list(enumerate(states))):
            acs_data = data_source.get_data(states=[state], download=False)
            features, label, group = ACSEmployment.df_to_numpy(acs_data) if task == "employment" else ACSIncome.df_to_numpy(acs_data)

            X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
                features, label, group, test_size=0.2, random_state=0)
            if task == "income": X_train, X_test = np.delete(X_train, [4,5], axis=1), np.delete(X_test, [4,5], axis=1) # removing cols occupation bc way too complicated to encode each occupation; removing each state of birth also bc way too much to encode

            X_test, y_test = X_test[:numdata], y_test[:numdata]

            train_data[state] = np.hstack((X_train, (1. - y_train).reshape(-1,1))) # combine features w/labels
            test_data[state] = np.hstack((X_test, (1. - y_test).reshape(-1,1)))

            rewards[state] = np.array([i for i in y_test]) # NO STOCHASTIC REWARDS
            individual_weights[state] = np.array([1/len(X_test) for _ in range(len(X_test))])

        # cobble together train/val data for all states
        train_set = np.vstack(list(train_data.values()))

        # make sure all factors start with 0 TODO export to utils file
        one_cols = [2,4,6,9,10,11,12,14,15] if task == "employment" else [1,3,6,7]
        for state in test_data:
            for col in one_cols:
                test_data[state][:,col] -= 1

        num_feat = 8 if task == "income" else 16

        # get ranges for each categorical variable, so that we can encode into dummies
        factor_cols = [i for i in range(2, num_feat)] if task == "employment" else [1,3,4,6,7]
        number_levels = {col : int(np.amax(train_set[:,col])+1) for col in factor_cols}

        # start making plots for each converged distribution
        methods = ["SPO", "CE"]
        loss_names = ['skim', 'acc', 'knapsack', 'fair', 'ce']

        # can be changed to a list of states of your choosing
        model_states = states
        eval_states = states

        for i, model_state in enumerate(model_states): # general 4 random models (rows in final vis)
            method = "SPO" if i % 2 == 0 else "CE"

            # define the model and cached weights
            model_path = "{}_{}_{}".format(model_state, method, task)
            model = torch.load("{}/{}.pt".format(model_path, model_path))
            all_weights = {}
            for loss_name in loss_names:
                with open("{}_optimized_10000_25_25/test_converged_weights_{}.pickle".format(model_path, loss_name), 'rb') as handle:
                    weights = pickle.load(handle)
                    all_weights[loss_name] = weights

            # create the plots
            for state in eval_states:
                for loss_name in loss_names:
                    # get distribution to use
                    to_use_weights = all_weights[loss_name][state]

                    # load in the data and get the necessary attributes
                    subset = test_data[state]
                    temp = np.array([subset])
                    X = torch.tensor(temp[:,:,:-1]).type(torch.float32)
                    y_pred_proba = model(X).detach()[0]

                    # sample and plot
                    selected_sample = np.vstack([y_pred_proba, subset[:,(1 if task == "employment" else 2)], subset[:,-2], subset[:,-1]]).T # preds, education, ???, label

                    # side-by-side plots
                    fig = plt.figure()
                    plt.subplot(1, 2, 1) # row 1, col 2 index 1

                    # plot by race
                    unique_races = np.unique(selected_sample[:,-2])
                    markers = [ "." , "^", "s", "*", "P", "D", "X", "<", ">"]

                    all_pos, all_neg = to_use_weights[np.where(selected_sample[:,-1] == 1)], to_use_weights[np.where(selected_sample[:,-1] == 1)]
                    vmin_pos, vmax_pos = np.amin(all_pos), np.amax(all_pos)
                    vmin_neg, vmax_neg = np.amin(all_neg), np.amax(all_neg)

                    for i, race in enumerate(unique_races):
                        race_set = selected_sample[selected_sample[:,-2] == race]
                        pos = race_set[race_set[:,-1] == 1]
                        plt.scatter(pos[:,1], pos[:,0], c=to_use_weights[np.where((selected_sample[:,-1] == 1) & (selected_sample[:,-2] == race))], marker=markers[i], vmin=vmin_pos, vmax=vmax_pos, s=200)
                    
                    plt.title("Positives")
                    plt.ylabel('Model Prediction')
                    plt.xlabel('Education')
                    plt.xlim(0,25)
                    cb = plt.colorbar()
                    plt.tight_layout()

                    plt.subplot(1, 2, 2) # index 2
                    for i, race in enumerate(unique_races):
                        race_set = selected_sample[selected_sample[:,-2] == race]
                        neg = race_set[race_set[:,-1] == 0]
                        plt.scatter(neg[:,1], neg[:,0], c=to_use_weights[np.where((selected_sample[:,-1] == 0) & (selected_sample[:,-2] == race))], marker=markers[i], vmin=vmin_neg, vmax=vmax_neg, s=200)
                    plt.title("Negatives")
                    plt.ylabel('Model Prediction')
                    plt.xlabel('Education')
                    plt.xlim(0,25)
                    cb = plt.colorbar()

                    name = loss_name if loss_name != "skim" else "top-k"

                    plt.suptitle(f"Joint dist {name} {state} {model_path}")
                    plt.tight_layout()

                    if not os.path.exists("../paper_visualizations"):
                        os.makedirs("../paper_visualizations")
                    if not os.path.exists(f"../paper_visualizations/{task}_classification"):
                        os.makedirs(f"../paper_visualizations/{task}_classification")

                    plt.savefig(f"../paper_visualizations/{task}_classification/{model_path}_{state}_{name}.pdf", format="pdf", bbox_inches='tight')

                    # # Clear the current axes.
                    plt.cla() 
                    # Clear the current figure.
                    plt.clf() 
                    # Closes all the figure windows.
                    plt.close('all')   
                    plt.close(fig)
                    gc.collect()

        print(f"done task {task}")