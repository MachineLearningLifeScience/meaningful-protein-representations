import pytorch_lightning as pl
import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
import numpy as np
from Bio import Phylo
from sklearn.cluster import KMeans
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
from scipy.special import softmax, erfinv
from copy import deepcopy
from itertools import chain


import os, sys
from .geoml.manifold import EmbeddedManifold, CubicSpline
from .geoml.discretized_manifold import DiscretizedManifold


# Mapping from amino acids to integers
aa1_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6,
                'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12,
                'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                'Y': 19, 'X':20, 'Z': 21, '-': 22}
aa1 = "ACDEFGHIKLMNPQRSTVWYXZ-"


# Entropy network
class translatedSigmoid(nn.Module):
    def __init__(self):
        super(translatedSigmoid, self).__init__()
        self.beta = nn.Parameter(torch.tensor([-3.5]))

    def forward(self, x):
        beta = torch.nn.functional.softplus(self.beta)
        alpha = -beta*(6.9077542789816375)
        return torch.sigmoid((x+alpha)/beta)


class DistNet(nn.Module):
    def __init__(self, dim, num_points):
        super().__init__()
        self.num_points = num_points
        self.points = nn.Parameter(torch.randn(num_points, dim),
                                  requires_grad=False)
        self.trans = translatedSigmoid()
        self.initialized = False

    def __dist2__(self, x):
        t1 = (x**2).sum(-1, keepdim=True)
        t2 = torch.transpose((self.points**2).sum(-1, keepdim=True), -1, -2)
        t3 = 2.0*torch.matmul(x, torch.transpose(self.points, -1, -2))
        return (t1 + t2 - t3).clamp(min=0.0)

    def forward(self, x):
        with torch.no_grad(): # To prevent backpropping back through to the dist points
            D2 = self.__dist2__(x) # |x|-by-|points|
            min_d = D2.min(dim=-1)[0] # smallest distance to clusters
            return self.trans(min_d)

    def kmeans_initializer(self, embeddings):
        km = KMeans(n_clusters=self.num_points).fit(embeddings)
        self.points.data = torch.tensor(km.cluster_centers_,
                                        device=self.points.device)
        self.initialized = True


class VAE(pl.LightningModule, EmbeddedManifold):
    def __init__(self, data, weights, perm, hparams, aa_weights=None):
        super().__init__()
        self.hparams = hparams

        self._train_encoder_only = False
        self.train_idx = int(data.shape[0] * self.hparams.train_fraction)
        self.val_idx = int(data.shape[0] * (self.hparams.train_fraction+self.hparams.val_fraction))
        self.data = data
        self.weights = weights
        self.perm = perm
        self.length = data.shape[1]

        zdim = self.hparams.zdim if "zdim" in self.hparams else 2
        
        self.encoder_architecture = [1500, 1500]#, 1500]
        self.decoder_architecture = [100, 500]#, 500]

        self.sparsity_prior_dictionary_size = 10
        self.group_prior_n_patterns = 4
        
        self.encoder = nn.Sequential(
                nn.Linear(self.length*len(aa1_to_index), self.encoder_architecture[0]),
                nn.ReLU(),
                nn.Linear(self.encoder_architecture[0], self.encoder_architecture[1]),
                nn.ReLU())

        self.encoder_mu = nn.Linear(self.encoder_architecture[-1], zdim)
        self.encoder_scale = nn.Sequential(nn.Linear(self.encoder_architecture[-1], zdim), nn.Softplus())

        if "sparsity_prior" in self.hparams and self.hparams.sparsity_prior:
        
            self.decoder = nn.Sequential(
                    nn.Linear(zdim, self.decoder_architecture[0]),
                    nn.ReLU(),
                    nn.Linear(self.decoder_architecture[0], self.decoder_architecture[1]),
                    nn.ReLU())

            self.W = nn.Linear(self.decoder_architecture[-1], self.length * self.sparsity_prior_dictionary_size, bias = False)
            self.C = nn.Linear(self.sparsity_prior_dictionary_size, len(aa1_to_index), bias = False)
            self.S = nn.Linear(self.decoder_architecture[-1] // self.group_prior_n_patterns, self.length, bias = False)
            self.bias = nn.Parameter(torch.ones(self.length, len(aa1_to_index)) * 0.1)
            
        else:
            self.decoder = nn.Sequential(
                    nn.Linear(zdim, self.decoder_architecture[0]),
                    nn.ReLU(),
                    nn.Linear(self.decoder_architecture[0], self.decoder_architecture[1]),
                    nn.ReLU(),
                    nn.Linear(self.decoder_architecture[-1], self.length*len(aa1_to_index)))

        self.aa_weights = aa_weights
        if "mask_out_gaps" in self.hparams and self.hparams.mask_out_gaps:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none', weight=aa_weights, ignore_index=aa1_to_index['-'])
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none', weight=aa_weights)

        self.prior = D.Independent(D.Normal(torch.zeros(zdim).to(self._device),
                                            torch.ones(zdim).to(self._device)), 1)

        # As a prior on S, we place a Normal prior on S prior to the sigmoid, which means that we effectively use
        # a logit-Normal distribution. We freely choose a sigma in this prior, and choose the mu so that
        # most of the probability mass (set by sparsity_S_prior_quantile) is below 0.0 - translating into sigmoid values
        # between zero and one
        self.sparsity_S_prior_quantile = 0.01
        self.sparsity_S_prior_sigma = 4.0
        # Note that this formula is the quantile formula for a Normal
        self.sparsity_S_prior_mu = np.sqrt(2.0) * self.sparsity_S_prior_sigma * erfinv(2.0 * self.sparsity_S_prior_quantile - 1.0)

        self.sparsity_prior = D.Normal(self.sparsity_S_prior_mu,
                                       self.sparsity_S_prior_sigma)
        
        self.distnet = DistNet(zdim, 200)
        self.switch = False
        self._prior = None

        # Setting to 1 to allow default behavior of beta=1
        self.warmup_step = 1
        
    @property
    def prior(self):
        if self._prior is None:
            zdim = self.hparams.zdim if "zdim" in self.hparams else 2
            self._prior =  D.Independent(D.Normal(torch.zeros(zdim).to(self.device),
                                                  torch.ones(zdim).to(self.device)), 1)
        return self._prior

    @prior.setter
    def prior(self, prior):
        self._prior = prior

    @property
    def beta(self):
        if self.training:
            self.warmup_step += 1
        return min([float(self.warmup_step/self.hparams.kl_warmup_steps), 1.0])
        
    def decode(self, z, as_probs=False, return_s=False):

        if "sparsity_prior" in self.hparams and self.hparams.sparsity_prior:
            h = self.decoder(z)
            S = torch.sigmoid(self.S.weight.transpose(0,1).repeat(self.group_prior_n_patterns, 1))

            # Apply dictionary factorization
            # W_out = torch.softplus(self.lambd) * (self.W @ self.C)
            W = self.W.weight.transpose(0,1).view(self.decoder_architecture[-1], self.length, self.sparsity_prior_dictionary_size)
            W_out = (W @ self.C.weight.transpose(0,1))

            # Apply group sparsity
            W_out = W_out * S.unsqueeze(-1)

            # Apply linear transformation
            recon = ((h @ W_out.view(self.decoder_architecture[-1], self.length*len(aa1_to_index))).view(*z.shape[:-1], self.length, -1) + self.bias.unsqueeze(0).unsqueeze(0)).transpose(-1, -2)
            
        else:
            recon = self.decoder(z).reshape(*z.shape[:-1], len(aa1_to_index), -1)

        if self.switch:
            if not self.distnet.initialized:
                train_embedding = [ ]
                for batch in self.train_dataloader():
                    batch = batch[0]
                    train_embedding.append(self.encoder_mu(self.encoder(
                            F.one_hot(batch.long().to(self._device), len(aa1_to_index)
                            ).float().reshape(batch.shape[0], -1))).cpu().detach().numpy())
                train_embedding = np.vstack(train_embedding)
                km = KMeans(n_clusters=self.distnet.num_points).fit(train_embedding)
                self.distnet.points.data = torch.tensor(km.cluster_centers_, device=self.device)
                self.distnet.initialized = True

            s = self.distnet(z).view(*z.shape[:-1], 1, 1)
            recon = (1-s) * recon + s*recon.mean()

        else:
            recon += torch.randn_like(recon)*1e-2

        if as_probs:
            recon = F.log_softmax(recon, dim=-2).exp()

        if return_s:
            return recon, s
        else:
            return recon

    def forward(self, x, n_samples=1):
        x = nn.functional.one_hot(x, len(aa1_to_index))
        h = self.encoder(x.float().reshape(x.shape[0], -1))

        q_dist = D.Independent(D.Normal(self.encoder_mu(h),
                                        self.encoder_scale(h) + 1e-4), 1)
        z_samples = q_dist.rsample(torch.Size([n_samples]))

        recon = self.decode(z_samples)
        return recon, q_dist, z_samples

    def embedding(self, x):
        x = nn.functional.one_hot(x, len(aa1_to_index))
        h = self.encoder(x.float().reshape(x.shape[0], -1))
        return self.encoder_mu(h)

    def _step(self, batch, batch_idx):
        x = batch[0].long()
        n_samples = 10 if self.hparams.iwae_bound else 1
        recon, q_dist, z_samples = self(x, n_samples=n_samples)
        recon_samples_in_batch_dim = recon.view(torch.Size([-1])+recon.shape[2:])
        x_repeated = x.unsqueeze(0).repeat(recon.shape[0], 1, 1, 1).view(torch.Size([-1])+x.shape[1:])
        if self.aa_weights is not None:
            log_prob_x = ((-self.loss_fn(recon_samples_in_batch_dim, x_repeated).view(torch.Size([-1, x.shape[0]])+recon.shape[3:])).sum(-1) / self.aa_weights[x].sum(dim=-1)) * x.shape[1]
        else:
            log_prob_x = -self.loss_fn(recon_samples_in_batch_dim, x_repeated).view(torch.Size([-1, x.shape[0]])+recon.shape[3:]).sum(dim=-1)

        # Prior contributions
        group_sparsity_loss = 0
        if self.hparams.sparsity_prior:
            log_prob_x += -self.hparams.sparsity_prior_lambda * self.sparsity_prior.log_prob(self.S.weight).sum()
            
        recon_loss = -log_prob_x.mean()
        kl_loss = D.kl_divergence(q_dist, self.prior).mean()
        
        if self.hparams.iwae_bound:
            # importance weighted autoencoder bound
            loss = -torch.mean(torch.logsumexp((log_prob_x + self.prior.log_prob(z_samples) - q_dist.log_prob(z_samples)), dim=0) - np.log(z_samples.shape[0]))
        else:
            # standard elbo
            loss = -torch.mean(log_prob_x + self.prior.log_prob(z_samples) - q_dist.log_prob(z_samples))

        acc = (recon.mean(dim=0).argmax(dim=1) == x)[x!=aa1_to_index['-']].float().mean()
        return loss, recon_loss, kl_loss, acc

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, acc = self._step(batch, batch_idx)

        self.log_dict({'train_loss': loss,
                       'train_recon': recon_loss,
                       'train_kl': kl_loss,
                       'train_acc': acc,
                       'beta': self.beta},
                      prog_bar=True,
                      logger=True)

        return loss

    def configure_optimizers(self):
        if self._train_encoder_only:
            print("only training encoder")
            return torch.optim.Adam(chain(self.encoder.parameters(),
                                          self.encoder_mu.parameters(),
                                          self.encoder_scale.parameters()), 
                                    lr=self.hparams.lr)
        else:        
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # def validation_step(self, batch, batch_idx):
    #     loss, recon_loss, kl_loss, acc = self._step(batch, batch_idx)

    #     self.log_dict({'val_loss': loss,
    #                    'val_recon': recon_loss,
    #                    'val_kl': kl_loss,
    #                    'val_acc': acc,
    #                    'beta': self.beta})

    # def test_step(self, batch, batch_idx):
    #     loss, recon_loss, kl_loss, acc = self._step(batch, batch_idx)

    #     self.log_dict({'test_loss': loss,
    #                    'test_recon': recon_loss,
    #                    'test_kl': kl_loss,
    #                    'test_acc': acc,
    #                    'beta': self.beta})

    def train_dataloader(self, labels=None):
        dataset = [self.data[self.perm[:self.train_idx]]]
        if labels is not None:
            dataset += [labels[self.perm[:self.train_idx]]]
        train_data = torch.utils.data.TensorDataset(*dataset)
        if self.weights is not None:
          weights_normalized = self.weights[self.perm[:self.train_idx]]
          weights_normalized /= weights_normalized.sum()
          sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_normalized, len(weights_normalized))
          return torch.utils.data.DataLoader(train_data, batch_size=self.hparams.bs, sampler = sampler)
        else:
          return torch.utils.data.DataLoader(train_data, batch_size=self.hparams.bs)

    def val_dataloader(self):
        val_data = torch.utils.data.TensorDataset(self.data[self.perm[self.train_idx:self.val_idx]])
        return torch.utils.data.DataLoader(val_data, batch_size=self.hparams.bs)

    def test_dataloader(self):
        test_data = torch.utils.data.TensorDataset(self.data[self.perm[self.val_idx:]])
        return torch.utils.data.DataLoader(test_data, batch_size=self.hparams.bs)
    
    def embed(self, points, jacobian=False):
        pass

    def curve_energy(self, curve):
        if curve.dim() == 2: curve.unsqueeze_(0) # BxNxd

        recon = self.decode(curve, as_probs=True) # BxNxFxS

        x = recon[:,:-1,:,:]; y = recon[:,1:,:,:]; # Bx(N-1)xFxS
        dt = torch.norm(curve[:,:-1,:] - curve[:,1:,:], p=2, dim=-1) # Bx(N-1)
        energy = (1-(x*y).sum(dim=2)).sum(dim=-1) # Bx(N-1) 
        return 2*(energy * dt).sum(dim=-1)

    def curve_length(self, curve):
        return torch.sqrt(self.curve_energy(curve))


def numeric_curve_optimizer(model, curve):
    optimizer = torch.optim.Adam([curve.parameters], lr=1e-2)
    alpha = torch.linspace(0, 1, 50).reshape((-1, 1))
    best_curve, best_loss = deepcopy(curve), float('inf')
    for i in range(10):
        optimizer.zero_grad()
        loss = model.curve_energy(curve(alpha)).sum()
        loss.backward()
        optimizer.step()
        grad_size = torch.max(torch.abs(curve.parameters.grad))
        if grad_size < 1e-3:
            break
        if loss.item() < best_loss:
            best_curve = deepcopy(curve)
            best_loss = loss.item()

    return best_curve


def get_hparams(args=None):

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-lr', default=1e-3, type=float)
    argparser.add_argument('-bs', default=16, type=int)
    argparser.add_argument('-gpu', default=1, nargs='?', type=str2bool)
    argparser.add_argument('-load_from', default='')
    argparser.add_argument('-epochs', default=20, type=int)
    argparser.add_argument('-zdim', default=2, type=int)
    argparser.add_argument('-seed', default=123, type=int)
    argparser.add_argument('-train_fraction', default=0.8, type=float)
    argparser.add_argument('-val_fraction', default=0.1, type=float)
    argparser.add_argument('-kl_warmup_steps', default=1, type=int)
    argparser.add_argument('-iwae_bound', default=0, nargs='?', type=str2bool)
    argparser.add_argument('-sparsity_prior', default=0, nargs='?', type=str2bool)
    argparser.add_argument('-mask_out_gaps', default=0, nargs='?', type=str2bool)
    argparser.add_argument('-sparsity_prior_lambda', default=1e-4, nargs='?', type=float)

    return argparser.parse_args(args)

    
