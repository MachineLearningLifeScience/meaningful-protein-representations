import pytorch_lightning as pl
import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
import numpy as np
from Bio import Phylo
from sklearn.cluster import KMeans
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
from scipy.special import softmax
from copy import deepcopy
from itertools import chain

import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "geoml"))
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
        self.beta = nn.Parameter(torch.tensor([3.5]))

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
        length = data.shape[1]
        
        self.encoder = nn.Sequential(
                nn.Linear(length*len(aa1_to_index), 1500),
                nn.ReLU(),
                nn.Linear(1500, 1500),
                nn.ReLU())

        self.encoder_mu = nn.Linear(1500, 2)
        self.encoder_scale = nn.Sequential(nn.Linear(1500, 2), nn.Softplus())

        self.decoder = nn.Sequential(
                nn.Linear(2, 100),
                nn.ReLU(),
                nn.Linear(100, 500),
                nn.ReLU(),
                nn.Linear(500, length*len(aa1_to_index)))

        # self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=aa1_to_index['-'])
        # self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.aa_weights = aa_weights
        self.loss_fn = nn.CrossEntropyLoss(reduction='none', weight=aa_weights)

        self.prior = D.Independent(D.Normal(torch.zeros(2).to(self._device),
                                            torch.ones(2).to(self._device)), 1)

        self.distnet = DistNet(2, 200)
        self.switch = False
        self._prior = None

        # Setting to 1 to allow default behavior of beta=1
        self.warmup_step = 1
        
    @property
    def prior(self):
        if self._prior is None:
            self._prior =  D.Independent(D.Normal(torch.zeros(2).to(self.device),
                                                  torch.ones(2).to(self.device)), 1)
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

    def forward(self, x):
        x = nn.functional.one_hot(x, len(aa1_to_index))
        h = self.encoder(x.float().reshape(x.shape[0], -1))

        q_dist = D.Independent(D.Normal(self.encoder_mu(h),
                                        self.encoder_scale(h) + 1e-4), 1)
        z = q_dist.rsample()

        recon = self.decode(z)
        return recon, q_dist

    def embedding(self, x):
        x = nn.functional.one_hot(x, len(aa1_to_index))
        h = self.encoder(x.float().reshape(x.shape[0], -1))
        return self.encoder_mu(h)

    def _step(self, batch, batch_idx):
        x = batch[0].long()
        recon, q_dist = self(x)
        #recon_loss = self.loss_fn(recon, x).sum(dim=-1).mean()
        recon_loss = (self.loss_fn(recon, x).sum(dim=0) / self.aa_weights[x].sum(dim=0)).sum()
        kl_loss = D.kl_divergence(q_dist, self.prior).mean()
        loss = recon_loss + self.beta * kl_loss
        acc = (recon.argmax(dim=1) == x)[x!=aa1_to_index['-']].float().mean()
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

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, acc = self._step(batch, batch_idx)

        self.log_dict({'val_loss': loss,
                       'val_recon': recon_loss,
                       'val_kl': kl_loss,
                       'val_acc': acc,
                       'beta': self.beta})

    def test_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, acc = self._step(batch, batch_idx)

        self.log_dict({'test_loss': loss,
                       'test_recon': recon_loss,
                       'test_kl': kl_loss,
                       'test_acc': acc,
                       'beta': self.beta})

    def train_dataloader(self):        
        train_data = torch.utils.data.TensorDataset(self.data[self.perm[:self.train_idx]])
        if self.weights is not None:
          sampler = torch.utils.data.sampler.WeightedRandomSampler(self.weights[self.perm[:self.train_idx]], len(self.weights[:self.train_idx]))
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

def curve_energy(model, curve, weight=0.0):
    if curve.dim() == 2:
        curve.unsqueeze_(0) # BxNxd

    recon, switch = model.decode(curve, as_probs=True, return_s=True) # BxNxFxS
    x = recon[:,:-1,:,:]; y = recon[:,1:,:,:];
    dt = torch.norm(curve[:,1:,:] - curve[:,:-1,:], p=2, dim=-1) # BxN
    energy = (2*(1 - (x * y).sum(dim=2))) # BxNxhparamsS
    energy = energy.mean(dim=-1) # BxN, use mean instead of sum for stability
    energy = (energy * dt) # BxN
    regulizer = (switch[:,:1,:,:] + switch[:,:-1,:,:]) / 2.0 # mean switch activation
    regulizer = weight*regulizer.view(energy.shape)*dt # BxN
    return (energy + regulizer).sum(dim=-1) # B


def numeric_curve_optimizer(model, curve):
    optimizer = torch.optim.Adam([curve.parameters], lr=1e-2)
    alpha = torch.linspace(0, 1, 50).reshape((-1, 1))
    best_curve, best_loss = deepcopy(curve), float('inf')
    for i in range(10):
        optimizer.zero_grad()
        loss = curve_energy(model, curve(alpha), 1.0).sum()
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
    
    argparser = ArgumentParser()
    argparser.add_argument('-lr', default=1e-3, type=float)
    argparser.add_argument('-bs', default=16, type=int)
    argparser.add_argument('-gpu', default=1, type=bool)
    argparser.add_argument('-load_from', default='')
    argparser.add_argument('-epochs', default=20, type=int)
    argparser.add_argument('-seed', default=123, type=int)
    argparser.add_argument('-train_fraction', default=0.8, type=float)
    argparser.add_argument('-val_fraction', default=0.1, type=float)
    argparser.add_argument('-kl_warmup_steps', default=1, type=int)

    return argparser.parse_args(args)

    
