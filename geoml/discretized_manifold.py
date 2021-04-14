#!/usr/bin/env python3
import torch
from geoml import *
import networkx as nx

class DiscretizedManifold:
    def __init__(self, model, grid, use_diagonals=False):
        """Approximate the latent space of a Manifold with a discrete grid.

        Inputs:
            model:  the Manifold to be approximated. This object should
                    implement the 'curve_length' function.

            grid:   a torch Tensor where the first dimension correspond
                    to the latent dimension of the manifold, and the
                    remaining are grid positions in a meshgrid format.
                    For example, a 2D manifold should be discretized by
                    a 2xNxM grid.
        """
        self.grid = grid
        self.G = nx.Graph()

        if grid.shape[0] != 2:
            raise Exception('Currently we only support 2D grids -- sorry!')

        # Add nodes to graph
        dim, xsize, ysize = grid.shape
        node_idx = lambda x, y: x*ysize + y
        self.G.add_nodes_from(range(xsize*ysize))
        #for x in range(xsize):
        #    for y in range(ysize):
        #        p = grid[:, x, y]
        #        self.G.add_node(node_idx(x, y))

        # add edges
        line = CubicSpline(begin=torch.zeros(1, dim), end=torch.ones(1, dim), num_nodes=2)
        t = torch.linspace(0, 1, 5)
        for x in range(xsize):
            for y in range(ysize):
                line.begin = grid[:, x, y].view(1, -1)
                n = node_idx(x, y)

                with torch.no_grad():
                    if x > 0:
                        line.end = grid[:, x-1, y].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x-1, y), weight=w)
                    if y > 0:
                        line.end = grid[:, x, y-1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x, y-1), weight=w)
                    if x < xsize-1:
                        line.end = grid[:, x+1, y].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x+1, y), weight=w)
                    if y < ysize-1:
                        line.end = grid[:, x, y+1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x, y+1), weight=w)
                    if use_diagonals and x > 0 and y > 0:
                        line.end = grid[:, x-1, y-1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x-1, y-1), weight=w)
                    if use_diagonals and x < xsize-1 and y > 0:
                        line.end = grid[:, x+1, y-1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x+1, y-1), weight=w)

    def grid_point(self, p):
        """Return the index of the nearest grid point.

        Input:
            p:      a torch Tensor corresponding to a latent point.

        Output:
            idx:    an integer correponding to the node index of
                    the nearest point on the grid.
        """
        return (self.grid.view(self.grid.shape[0], -1) - p.reshape(-1, 1)).pow(2).sum(dim=0).argmin().item()

    def shortest_path(self, p1, p2):
        """Compute the shortest path on the discretized manifold.

        Inputs:
            p1:     a torch Tensor corresponding to one latent point.

            p2:     a torch Tensor corresponding to another latent point.

        Outputs:
            curve:  a DiscreteCurve forming the shortest path from p1 to p2.

            dist:   a scalar indicating the length of the shortest curve.
        """
        idx1 = self.grid_point(p1)
        idx2 = self.grid_point(p2)
        path = nx.shortest_path(self.G, source=idx1, target=idx2, weight='weight') # list with N elements
        coordinates = self.grid.view(self.grid.shape[0], -1)[:, path] # (dim)xN
        N = len(path)
        curve = DiscreteCurve(begin=coordinates[:, 0], end=coordinates[:, -1], num_nodes=N)
        with torch.no_grad():
            curve.parameters[:, :] = coordinates[:, 1:-1].t()
        dist = 0
        for i in range(N-1):
            dist += self.G.edges[path[i], path[i+1]]['weight']
        return curve, dist

    def connecting_geodesic(self, p1, p2, curve=None):
        """Compute the shortest path on the discretized manifold and fit
        a smooth curve to the resulting discrete curve.

        Inputs:
            p1:     a torch Tensor corresponding to one latent point.

            p2:     a torch Tensor corresponding to another latent point.

        Optional input:
            curve:  a curve that should be fitted to the discrete graph
                    geodesic. By default this is None and a CubicSpline
                    with default paramaters will be constructed.

        Outputs:
            curve:  a smooth curve forming the shortest path from p1 to p2.
                    By default the curve is a CubicSpline with its default
                    parameters; this can be changed through the optional
                    curve input.
        """
        device = p1.device
        idx1 = self.grid_point(p1)
        idx2 = self.grid_point(p2)
        path = nx.shortest_path(self.G, source=idx1, target=idx2, weight='weight') # list with N elements
        weights = [self.G.edges[path[k], path[k+1]]['weight'] for k in range(len(path)-1)]
        coordinates = (self.grid.view(self.grid.shape[0], -1)[:, path[1:-1]]).t() # Nx(dim)
        t = torch.tensor(weights[:-1], device=device).cumsum(dim=0) / sum(weights)

        if curve is None:
            curve = CubicSpline(p1, p2, num_nodes=10)
        else:
            curve.begin = p1
            curve.end = p2

        curve.fit(t, coordinates)

        return curve
