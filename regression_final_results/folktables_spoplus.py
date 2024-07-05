#!/usr/bin/env python
# coding: utf-8
"""
SPO+ Loss function
"""
import numpy as np
import torch
from torch.autograd import Function
from folktables_abcmodule import optModule

# dp method of solving knapsack problem and its argmax over a 2d array of weights and values, given capacity and length of arr
def _solve_in_pass(wt, val, W, n):
    val = np.maximum(0, 70000. - val)
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
        res = knapsack(wt, val, W, n)
        return res
    sol, obj = [], []
    for wt_temp, val_temp in zip(wt, val):
        temp_obj, temp_sol = helperSolve(wt_temp, val_temp, W, n)
        temp_sol = np.array([x - 1 for x in temp_sol])
        assert(len(temp_sol) == 0 or np.amin(temp_sol) >= 0)
        new_sol = np.zeros(n)
        new_sol[list(temp_sol)] += 1.
        sol.append(new_sol)
        obj.append(temp_obj)
    return sol, obj

class SPOPlus(optModule):
    """
    An autograd module for SPO+ Loss, as a surrogate loss function of SPO Loss,
    which measures the decision error of the optimization problem.

    For SPO/SPO+ Loss, the objective function is linear and constraints are
    known and fixed, but the cost vector needs to be predicted from contextual
    data.

    The SPO+ Loss is convex with subgradient. Thus, it allows us to design an
    algorithm based on stochastic gradient descent.

    Reference: <https://doi.org/10.1287/mnsc.2020.3922>
    """

    def __init__(self, dataset=None):
        """
        Args:
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(dataset)
        # build carterion
        self.spop = SPOPlusFunc()

    def forward(self, pred_cost, true_cost, true_sol, true_obj, weights, reduction="mean"):
        """
        Forward pass
        """
        loss = self.spop.apply(pred_cost, true_cost, true_sol, true_obj, weights)
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss


class SPOPlusFunc(Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(ctx, pred_cost, true_cost, true_sol, true_obj, weights):
        """
        Forward pass for SPO+

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            solve_ratio (float): the ratio of new solutions computed during training
            module (optModule): SPOPlus modeul

        Returns:
            torch.tensor: SPO+ loss
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        c = true_cost.detach().to("cpu").numpy()
        w = true_sol.detach().to("cpu").numpy()
        z = true_obj.detach().to("cpu").numpy()
        weights = weights.detach().to("cpu").numpy().astype(np.int32)
        sol, obj = _solve_in_pass(weights, 2*cp-c, 10*len(weights[0]), len(weights[0]))

        # calculate loss
        loss = []
        for i in range(len(cp)):
            loss.append(- obj[i] + 2 * np.dot(cp[i], w[i]) - z[i])
        # sense
        # loss = - np.array(loss)
        loss = np.array(loss)

        # convert to tensor
        loss = torch.FloatTensor(loss).to(device)
        sol = np.array(sol)
        sol = torch.FloatTensor(sol).to(device)
        # save solutions
        ctx.save_for_backward(true_sol, sol)
        # add other objects to ctx
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO+
        """
        w, wq = ctx.saved_tensors
        grad = 2 * (wq - w)
        return grad_output * grad, None, None, None, None, None, None, None, None