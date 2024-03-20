import numpy as np
import torch


def find_cutpoints2tensor(discretization, ord_num, alpha, beta, gamma):
    alpha_star = alpha + gamma
    beta_star = beta + gamma

    if discretization == "SID":
        cutpoints = [
            np.exp(np.log(alpha_star) + ((np.log(beta_star) - np.log(alpha_star)) * float(b + 1) / ord_num))
            for b in range(ord_num)]
    elif discretization == "UD":
        cutpoints = [alpha_star + (beta_star - alpha_star) * (float(b + 1) / ord_num) for b in range(ord_num)]
    else:
        cutpoints = np.sort(np.random.uniform(low=alpha_star, high=beta_star, size=ord_num))

    cutpoints = torch.tensor(cutpoints, requires_grad=False) - gamma
    t0s = torch.cat((torch.tensor(alpha).view(-1), cutpoints), dim=0)
    t1s = torch.cat((cutpoints, torch.tensor(beta+1).view(-1)), dim=0)
    bin_values = (t0s + t1s) / 2
    return cutpoints, t0s, t1s, bin_values