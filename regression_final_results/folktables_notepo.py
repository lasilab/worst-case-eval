import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from folktables2.folktables import ACSDataSource, ACSEmployment, ACSIncome
from sklearn.model_selection import train_test_split
import warnings
import argparse
import os
import json
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from tqdm import tqdm
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

if __name__ == "__main__":
    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"][:5]

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("filename")
    parser.add_argument("state")
    args = parser.parse_args()
    task = args.task
    task = "income"
    model_state = args.state
    filename = args.filename
    # print("solving task {}; saving into file {}".format(task, filename))
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
    y_test = 0. - y_test
    one_cols = [2,4,6,9,10,11,12,14,15] if task == "employment" else [1,3,6,7]
    for col in one_cols:
        X_al[:,col] -= 1
        X_test[:,col] -= 1
        assert(np.amin(X_al[:,col]) == 0)

    num_feat = 8 if task == "income" else 16

    predmodel = NeuralNetworkIncome(max_min, num_feat, 'acc')

    # set optimizer
    optimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-2)
    # build dataset
    tensor_x = torch.Tensor(X_al)
    tensor_y = torch.Tensor(y_al)
    dataset = TensorDataset(tensor_x, tensor_y)
    # get data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    loss_fn = torch.nn.MSELoss()
    losses = []
    accs = []

    # write metadata to json
    json_obj = {
        "states": states, # number of data/samples we take
        "method": "acc", # not using spo loss
        'task': task
    }
    with open("{}/metadata.json".format(filename), 'w') as f:
        json.dump(json_obj, f)

    # training
    num_epochs = 200
    for epoch in tqdm(range(num_epochs)):
        for data in dataloader:
            x, y = data
            # forward pass
            cp = predmodel(torch.unsqueeze(x, 0))[0]
            loss = loss_fn(cp, y)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # accuracy calculation
        predmodel.eval()
        preds = predmodel(torch.unsqueeze(torch.tensor(X_al),0).type(torch.float32))[0]
        loss = loss_fn(preds, torch.tensor(y_al).type(torch.float32))
        losses.append(float(loss))

        test_preds = predmodel(torch.unsqueeze(torch.tensor(X_test), 0).type(torch.float32))[0] # TODO predict + save results on the test set

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            torch.save(predmodel, "{}/{}.pt".format(filename, filename))

        plt.plot([i for i in range(len(losses))], losses)
        plt.title("Training Loss: basic logit")
        plt.savefig("{}/loss_curve.png".format(filename))
        plt.close('all')

print("saved to {}".format(filename))
