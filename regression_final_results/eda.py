'''
Over a list of model states and test states, creates joint visualizations of the test sets of the test states,
when passed through our framework and a model specified via the model state(s).
'''

import random
import gc
import pickle
import numpy as np
import torch
import torch.nn as nn
from folktables2.folktables import ACSDataSource, ACSEmployment, ACSIncome
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from matplotlib import pyplot as plt
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

plt.rc("axes",titlesize=20)
plt.rc("axes",labelsize=20)
plt.rc("font",size=14)

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

if __name__ == "__main__":
    # load in cached weights
    setting = "income"
    numdata = 25
    loss_fns = ["mse", "top-k", "knapsack", "util", "fair"]

    # load in data
    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"][:5]

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

        test_data[state] = np.hstack((X_test, (0. - y_test).reshape(-1,1)))

    one_cols = [2,4,6,9,10,11,12,14,15] if setting == "employment" else [1,3,6,7]
    for state in test_data:
        for col in one_cols:
            test_data[state][:,col] -= 1
    
    # can be replaced with lists of states of user's choosing
    model_states = states
    data_states = states

    for i, model_state in enumerate(model_states):
        model_type = "SPO" if i % 2 == 0 else "CE"
        # run model on the data
        model_path = f"{model_state}_{model_type}_{setting}"
        worst_weights = {}
        for loss_fn in loss_fns:
            with open(f"{model_path}_optimized_10000_{numdata}_{numdata}/test_converged_weights_{loss_fn}.pickle", 'rb') as handle:
                worst_weights[loss_fn] = pickle.load(handle)
        model = torch.load("{}/{}.pt".format(model_path, model_path))

        for state in data_states:
            subset = test_data[state]
            temp = np.array([subset])
            X = torch.tensor(temp[:,:,:-1]).type(torch.float32)
            y_pred_proba = model(X).detach()[0]
            if "SPO" in model_path: y_pred_proba = np.maximum(0., 70000. - y_pred_proba)
            subset = np.hstack([subset, np.expand_dims(y_pred_proba, -1)])
            unique_races = np.unique(subset[:,-3])
            markers = [ "." , "^", "s", "*", "P", "D", "X", "<", ">"]
        
            for loss_fn in loss_fns:
                weights = worst_weights[loss_fn][state]
                new_set = np.hstack([subset, np.expand_dims(weights, -1)])
                vmin, vmax = np.amin(weights), np.amax(weights)
                for i, race in enumerate(unique_races):
                    race_subset = new_set[new_set[:,-4] == race]
                    plt.scatter(race_subset[:,-3], race_subset[:,-2], c=race_subset[:,-1], vmin=vmin, vmax=vmax, marker=markers[i], s=200) # plot label against pred, colored by weight
                plt.colorbar()
                plt.xlabel("True Income")
                plt.ylabel("Model Prediction")
                plt.suptitle(f"Joint dist {loss_fn} {state} {model_path}")
                plt.tight_layout()

                loss_name = loss_fn if loss_fn != "skim" else "top-k"

                if not os.path.exists("../paper_visualizations"):
                    os.makedirs("../paper_visualizations")
                if not os.path.exists("../paper_visualizations/regression"):
                    os.makedirs("../paper_visualizations/regression")

                plt.savefig(f"../paper_visualizations/regression/{model_path}_{state}_{loss_name}.pdf", format="pdf", bbox_inches='tight')
                # # Clear the current axes.
                plt.cla() 
                # Clear the current figure.
                plt.clf() 
                # Closes all the figure windows.
                plt.close('all')   
                gc.collect()
