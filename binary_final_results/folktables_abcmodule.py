#!/usr/bin/env python
# coding: utf-8
"""
Abstract autograd optimization module
"""

from abc import abstractmethod

import numpy as np
from torch import nn


class optModule(nn.Module):
    """
        An abstract module for the learning to rank losses, which measure the difference in how the predicted cost
        vector and the true cost vector rank a pool of feasible solutions.
    """
    def __init__(self, dataset):
        """
        Args:
            dataset (None/Dataset): the training data
        """
        super().__init__()
        # solution pool
        self.solpool = np.unique(dataset.sols.copy(), axis=0) # remove duplicate

    @abstractmethod
    def forward(self, pred_cost, true_cost, reduction="mean"):
        """
        Forward pass
        """
        # convert tensor
        pass