#!/usr/bin/env python
# coding: utf-8
"""
optDataset class based on PyTorch Dataset
"""


import numpy as np
import torch
from torch.utils.data import Dataset
import time
from tqdm import tqdm

class optDataset(Dataset):
    """
    This class is Torch Dataset for optimization problems.

    Attributes:
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        objs (np.ndarray): Optimal objective values
    """

    def __init__(self, feats, costs, weights, get_sol_obj_fn):
        """

        Args:
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
        """

        # data
        self.feats = feats # raw folktables data
        self.costs = costs # raw labels
        self.weights = weights # weight of each individual in each sample: here it's education level
        self.get_sol_obj_fn = get_sol_obj_fn # given the labels and weights for a group of people, return optimal treatment/rewards
        # find optimal solutions
        self.sols, self.objs = self._getSols()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols = []
        objs = []
        print("Optimizing for optDataset...")
        time.sleep(1)
        for c, w in tqdm(zip(self.costs, self.weights)):
            sol, obj = self._solve(c, w)
            sols.append(sol)
            objs.append([obj])
        return np.array(sols), np.array(objs)

    def _solve(self, cost, weight):
        """
        A method to solve optimization problem to get an optimal solution with given cost and weight

        Args:
            cost (np.ndarray): cost of objective function

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        sol, obj = self.get_sol_obj_fn(cost, weight)
        return sol, obj

    def __len__(self):
        """
        A method to get data size

        Returns:
            int: the number of optimization problems
        """
        return len(self.costs)

    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:
            tuple: data features (torch.tensor), costs (torch.tensor), optimal solutions (torch.tensor) and objective values (torch.tensor)
        """
        return (
            torch.FloatTensor(self.feats[index]),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
            torch.FloatTensor(self.weights[index])
        )