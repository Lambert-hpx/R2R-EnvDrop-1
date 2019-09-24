import numpy as np
import torch
from torch.autograd import Variable

EPS = 1e-8

class Categorical():
    def __init__(self, prob):
        self.prob = prob
        self.dim = self.prob.size()[-1]
        self.bs = prob.size()[0]
        self._mle = None

    def log_likelihood(self, x):
        if x.size()[-1] != self.prob.size()[-1]:
            prob = torch.gather(self.prob, -1, x).squeeze()
            return torch.log(prob + EPS)
        else:
            return torch.log(torch.sum(self.prob * x, -1) + EPS)

    def entropy(self):
        return -(self.prob * torch.log(self.prob + EPS)).sum(-1)

    def log_likelihood_ratio(self, x, new_dist):
        ll_new = new_dist.log_likelihood(x)
        ll_old = self.log_likelihood(x)
        return torch.exp(ll_new - ll_old)

    def kl(self, dist):
        # Compute D_KL(self || dist)
        return torch.sum(dist.prob * (torch.log(dist.prob + EPS) - torch.log(self.prob + EPS)), -1)

    def sample(self, deterministic=False):
        if deterministic:
            return self.prob.max(-1)[1].unsqueeze(-1)
        else:
            return torch.multinomial(self.prob, 1)

    def compute_mle(self):
        onehot = np.zeros(self.prob.size())
        onehot[np.arange(0, self.bs), get_numpy(torch.max(self.prob, -1)[1]).astype(np.int32)] = 1
        return np_to_var(onehot)

    def combine(self, dist_lst, func=torch.stack, axis=0):
        self.prob = func([dist.prob for dist in dist_lst], axis)
        return self

    def detach(self):
        return Categorical(self.prob.detach())

    def reshape(self, new_shape):
        self.prob = self.prob.view(*new_shape)
        return self

    @property
    def mle(self):
        if self._mle is None:
            self._mle = self.compute_mle()
        return self._mle

class Normal():
    def __init__(self, mean, log_var):
        self.mean = mean
        self.dim = self.mean.size()[-1]
        self.log_var = log_var


    def log_likelihood(self, x):
        # Assumes x is batch size by feature dim
        # Returns log_likehood for each point in batch
        zs = (x - self.mean) / torch.exp(self.log_var)

        return -torch.sum(self.log_var, -1) - \
               0.5 * torch.sum(torch.pow(zs, 2), -1) -\
               0.5 * self.dim * np.log(2*np.pi)

    def log_likelihood_ratio(self, x, new_dist):
        ll_new = new_dist.log_likelihood(x)
        ll_old = self.log_likelihood(x)
        return torch.exp(ll_new - ll_old)

    def entropy(self):
        return torch.sum(self.log_var + np.log(np.sqrt(2 * np.pi * np.e)), -1)

    def kl(self, new_dist):
        # Compute D_KL(self || dist)
        old_var = torch.exp(self.log_var)
        new_var = torch.exp(new_dist.log_var)
        numerator = torch.pow(self.mean - new_dist.mean, 2) + \
                    torch.pow(old_var, 2) - torch.pow(new_var, 2)
        denominator = 2 * torch.pow(new_var, 2) + 1e-8

        return torch.sum(
            numerator / denominator + new_dist.log_var - self.log_var, -1
        )


    def sample(self, deterministic=False):
        if deterministic:
            return self.mean
        else:
            return Variable(torch.randn(self.mean.size())).cuda() * torch.exp(self.log_var) + self.mean


    def combine(self, dist_lst, func=torch.stack, axis=0):
        self.mean = func([dist.mean for dist in dist_lst], axis)
        #self.var = func([dist.var for dist in dist_lst], axis)
        self.log_var = func([dist.log_var for dist in dist_lst], axis)
        self.dim = self.mean.size()[-1]
        return self

    def detach(self):
        return Normal(self.mean.detach(), self.log_var.detach())

    def reshape(self, new_shape):
        return Normal(self.mean.view(*new_shape), self.log_var.view(*new_shape))

    @property
    def mle(self):
        return self.mean
