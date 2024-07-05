import numpy as np
import torch
from torch import nn

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
    # s_m_2 = b / m - (w_m ** 2)
    # beta_sq = (2*rho + n - (n ** 2) / m) / ((n ** 2) * m * s_m_2)
    # beta_sq = (2*rho + n - (n ** 2) / m) / ((n ** 2) * s_m_2_times_m_2 / m)
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

# TODO make separate models for both income and employment prediction

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

class NeuralNetworkEmployment(nn.Module): # TODO make work with embeddings
    def __init__(self, train_set, num_feat):
        super().__init__()

        # create embeddings
        self.cat_cols = [i for i in range(2, 16)]
        self.embedding_size_dict = {col : int(np.amax(train_set[:,col]) + 1) for col in self.cat_cols}
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

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_feat - len(self.cat_cols) + self.total_embed_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10,1)
        )        
    # make predictions
    def forward(self, x):
        embedded_vector = x[:,:,:2]
        for col in range(2, self.num_feat): # TODO make not hacky (range hardcoded for employment)
            # we know that this col is categorical; map each row to its embedding and concat
            embeddings = self.embeddings[str(col)](x[:,:,col].type(torch.int64))
            embedded_vector = torch.cat((embedded_vector, embeddings), dim=2)
        out = torch.squeeze(torch.sigmoid(self.linear_relu_stack(embedded_vector)), -1)
        return out[0]
    
class LogisticRegressionIncome(nn.Module):    
    # build the constructor
    def __init__(self, train_set, num_feat, task):
        super(LogisticRegressionIncome, self).__init__()

        # create embeddings
        self.cat_cols = [1,3,4,6,7]
        self.embedding_size_dict = {col : int(np.amax(train_set[:,col]) + 1) for col in self.cat_cols}
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

        self.linear = torch.nn.Linear(self.num_feat - len(self.cat_cols) + self.total_embed_dim, 1)
        
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
        for col in range(1, self.num_feat): # TODO make not hacky (range hardcoded for employment)
            if col not in self.cat_cols:
                embedded_vector = torch.cat((embedded_vector, torch.unsqueeze(x[:,:,col], -1)), dim=2)
            else:
                # we know that this col is categorical; map each row to its embedding and concat
                embeddings = self.embeddings[str(col)](x[:,:,col].type(torch.int64))
                embedded_vector = torch.cat((embedded_vector, embeddings), dim=2)
        out = torch.squeeze(torch.sigmoid(self.linear_relu_stack(embedded_vector)), -1)
        return out
    
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
