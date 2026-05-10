import math

import torch
import torch.nn as nn

import numpy as np

import os
import json

from scipy.special import expit, logit
from scipy.optimize import minimize

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class SinusoidalEncoder(nn.Module):
    def __init__(self, data=None, dim=None, max_period=10000, dim_scale=0.25):
        super().__init__()
        assert data is not None or dim is not None

        if dim is None:
            dim = len(data)**dim_scale

        half = dim // 2
        self.freqs = nn.Parameter(
            torch.exp(-math.log(max_period) * 
                torch.arange(start=0, end=half, dtype=torch.float) / half
            ), requires_grad=False)

        self.res = dim % 2

    def forward(self, timesteps):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.FloatTensor(timesteps)
        timesteps = timesteps.to(self.freqs.device)

        args = (timesteps[..., None] * self.freqs).view(timesteps.shape[0], -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return torch.cat([embedding, torch.zeros_like(embedding[:, :self.res])], dim=-1)

def get_lambda_rate(decay_iters, warmup_iters=100, decay_rate=0.1):
    def f(iteration):
        rate = 1.0 if iteration > warmup_iters else iteration / warmup_iters
        return rate * decay_rate ** (iteration // decay_iters)
    return f

def load_config(data_name):
    with open(os.path.join('config/scenarios', data_name) + '.json') as f:
        return json.load(f)

def load_optimal_hparams(data_name, configuration_name):
    path = os.path.join('results/hparams', data_name, configuration_name, 'hparams.json')
    with open(path) as f:
        return json.load(f)

    
def solve_one_dimensional_submodel(y_hat, y, H):
    '''fit univariate logistic regression: y ~ reg.predict(X) + eps with weight H
    
    Parameters
    ----------
    logit_y_hat: n dimensional vector
        logit of predicted probability of y = 1
    y: n dimensional vector
        labels for each sample
    H: n dimensional vector
        weight for logistic regression

    Returns
    -------
    eps: `float`
        translation which minimize the logistic loss    
    '''
    logit_y_hat = logit(y_hat)

    def _safelog(x, delta=1e-8):
        return np.log(np.clip(x, delta, np.inf))

    def _loss(eps):
        '''weighted logistic loss function (binary cross entropy loss)'''
        s = expit(logit_y_hat + eps)
        return -np.mean(H * (y * _safelog(s) + (1 - y) * _safelog(1 - s)))
    
    def _jac(eps):
        '''gradient of the loss function'''
        s = expit(logit_y_hat + eps)
        return -np.mean(H * (y - s))
    
    return minimize(_loss, 0, method='L-BFGS-B', jac=_jac, tol=1e-14).x[0]

def solve_one_dimensional_submodel_regularized(y_hat, y, H, lmbda=5e-3):
    '''fit univariate logistic regression: y ~ reg.predict(X) + eps with weight H
    
    Parameters
    ----------
    logit_y_hat: n dimensional vector
        logit of predicted probability of y = 1
    y: n dimensional vector
        labels for each sample
    H: n dimensional vector
        weight for logistic regression
    lmbda: `float`
        L1 regularization parameter

    Returns
    -------
    eps: `float`
        translation which minimize the logistic loss    
    '''
    logit_y_hat = logit(y_hat)

    def _safelog(x, delta=1e-8):
        return np.log(np.clip(x, delta, np.inf))

    def _loss(eps):
        '''weighted logistic loss function (binary cross entropy loss)'''
        s = expit(logit_y_hat + eps)
        # Negative log-likelihood + L1 penalty.
        nll = -np.mean(H * (y * _safelog(s) + (1 - y) * _safelog(1 - s)))
        return nll + lmbda * np.abs(eps)
    
    def _jac(eps):
        '''gradient of the loss function'''
        s = expit(logit_y_hat + eps)
        # Gradient of (NLL + lmbda * |eps|). For L1, we use the subgradient.
        nll_grad = -np.mean(H * (y - s))
        return nll_grad + lmbda * np.sign(eps)
    
    return minimize(_loss, 0, method='L-BFGS-B', jac=_jac, tol=1e-14).x[0]

def solve_one_dimensional_submodel_CATE(y_hat_cf1, y_cf1, H_cf1, y_hat_cf2, y_cf2, H_cf2):
    logit_y_hat_cf1 = logit(y_hat_cf1)
    logit_y_hat_cf2 = logit(y_hat_cf2)

    def _safelog(x, delta=1e-8):
        return np.log(np.clip(x, delta, np.inf))

    def _loss(eps):
        '''weighted logistic loss function (binary cross entropy loss)'''
        s_cf1 = expit(logit_y_hat_cf1 + eps)
        s_cf2 = expit(logit_y_hat_cf2 - eps)
        return -np.mean(H_cf1 * (y_cf1 * _safelog(s_cf1) + (1 - y_cf1) * _safelog(1 - s_cf1)) + H_cf2 * (y_cf2 * _safelog(s_cf2) + (1 - y_cf2) * _safelog(1 - s_cf2)))
    
    def _jac(eps):
        '''gradient of the loss function'''
        s_cf1 = expit(logit_y_hat_cf1 + eps)
        s_cf2 = expit(logit_y_hat_cf2 - eps)
        return -np.mean(H_cf1 * (y_cf1 - s_cf1) - H_cf2 * (y_cf2 - s_cf2))
    
    return minimize(_loss, 0, method='L-BFGS-B', jac=_jac, tol=1e-14).x[0]
    
    

def get_torch_device(args):
    if args['use_cpu']:
        return 'cpu'
    elif torch.cuda.is_available():
        return 'cuda'
    # elif torch.backends.mps.is_available():
    #     return 'mps'
    
    return 'cpu'