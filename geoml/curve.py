#!/usr/bin/env python3
import torch

class BasicCurve:
    def plot(self, t0=0, t1=1, N=100, *args, **kwargs):
        with torch.no_grad():
            import torchplot as plt
            t = torch.linspace(t0, t1, N, device=self.device)
            points = self(t) # NxD or BxNxD
            if len(points.shape) == 2:
                points.unsqueeze_(0) # 1xNxD
            if points.shape[-1] == 1:
                for b in range(points.shape[0]):
                    plt.plot(t, points[b], *args, **kwargs)
            elif points.shape[-1] == 2:
                for b in range(points.shape[0]):
                    plt.plot(points[b, :, 0], points[b, :, 1], *args, **kwargs)
            else:
                print('BasicCurve.plot: plotting is only supported in 1D and 2D')

    def euclidean_length(self, t0=0, t1=1, N=100):
        t = torch.linspace(t0, t1, N)
        points = self(t) # NxD or BxNxD
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)
        delta = points[:, 1:] - points[:, :-1] # Bx(N-1)xD
        energies = (delta**2).sum(dim=2) # Bx(N-1)
        lengths = energies.sqrt().sum(dim=1) # B
        return lengths



class DiscreteCurve(BasicCurve):
    def __init__(self, begin, end, num_nodes=5, device=None, requires_grad=True):
        self.device = begin.device if device is None else device
        self.t = torch.linspace(0, 1, num_nodes, dtype=begin.dtype)[1:-1].view((-1, 1)) # (num_nodes-2)x1
        self.num_nodes = num_nodes
        self.begin = begin.reshape((1, -1))
        self.end = end.reshape((1, -1))
        self.parameters = (self.t.mm(self.end) + (1-self.t).mm(self.begin)).requires_grad_() # (num_nodes-2)xD

    def __call__(self, t):
        tflat = t.flatten()
        start_nodes = torch.cat((self.begin, self.parameters))  # (num_edges)xD
        end_nodes   = torch.cat((self.parameters, self.end))    # (num_edges)xD
        num_edges, D = start_nodes.shape
        t0 = torch.cat((torch.zeros(1, 1, dtype=self.t.dtype),
                        self.t.reshape((-1, 1)),
                        torch.ones(1, 1, dtype=self.t.dtype)))
        a = (end_nodes - start_nodes) / (t0[1:] - t0[:-1]).expand(-1, D) # (num_edges)xD
        b = start_nodes - a * t0[:-1].expand(-1, D) # (num_edges)xD

        #idx = torch.tensor([torch.nonzero(tflat[i] <= t0.flatten()[1:])[0] for i in range(t.numel())]) # use this if nodes are not equi-distant
        idx = torch.floor(t.flatten() * num_edges).clamp(min=0, max=num_edges-1).long() # use this if nodes are equi-distant
        tt = t.reshape((-1, 1)).expand(-1, D)
        result = a[idx]*tt + b[idx] # NxD
        return result

    #def reparam(self):
    #    with torch.no_grad():
    #        start_nodes = torch.cat((self.begin, self.parameters))  # (num_edges)xD
    #        end_nodes   = torch.cat((self.parameters, self.end))    # (num_edges)xD
    #        edge_len = (end_nodes - start_nodes).norm(dim=1) # (num_edges)
    #        cs_len =  edge_len.cumsum(dim=0) # (num_edges)
    #        cs_len /= cs_len[-1]
    #
    #        L = DiscreteCurve(torch.zeros(1), torch.ones(1), num_nodes=self.num_nodes, requires_grad=False)
    #        L.t = cs_len[:-1].reshape((-1, 1))
    #        new_t = L(torch.linspace(0, 1, self.num_nodes)[1:-1])
    #        self.parameters[:] = self(new_t)


class QuadraticCurve(BasicCurve):
    def __init__(self, begin, end, device=None, requires_grad=True):
        self.device = begin.device if device is None else device

        #begin # D or 1xD or BxD
        if len(begin) is 1 or begin.shape[0] is 1:
            self.begin = begin.detach().view((1, -1)) # 1xD
        else:
            self.begin = begin.detach() # BxD
        if len(end) is 1 or end.shape[0] is 1:
            self.end = end.detach().view((1, -1)) # 1xD
        else:
            self.end = end.detach() # BxD

        self.delta = end - begin # BxD
        self.parameters = torch.zeros(self.delta.shape, dtype=begin.dtype,
                                      device=self.device, requires_grad=requires_grad) # BxD
        self.c = begin # BxD

    def __call__(self, t):
        b = self.delta - self.parameters # BxD
        tt = t.view(1, -1, 1).repeat(b.shape[0], 1, 1) # Bx1|t|x1
        tt2 = tt**2 # Bx|t|x1
        return tt2.bmm(self.parameters.unsqueeze(1)) + tt.bmm(b.unsqueeze(1)) + self.c.unsqueeze(1) # Bx|t|xD


class CubicSpline(BasicCurve):
    # Compute cubic spline basis with end-points (0, 0) and (1, 0)
    def compute_basis(self, num_edges, device=None):
        with torch.no_grad():
            # set up constraints
            t = torch.linspace(0, 1, num_edges+1, dtype=self.begin.dtype)[1:-1]

            end_points = torch.zeros(2, 4*num_edges, dtype=self.begin.dtype, device=device)
            end_points[0, 0] = 1.0
            end_points[1, -4:] = 1.0

            zeroth = torch.zeros(num_edges-1, 4*num_edges, dtype=self.begin.dtype, device=device)
            for i in range(num_edges-1):
                si = 4*i # start index
                fill = torch.tensor([1.0, t[i], t[i]**2, t[i]**3], dtype=self.begin.dtype)
                zeroth[i, si:(si+4)] = fill
                zeroth[i, (si+4):(si+8)] = -fill

            first = torch.zeros(num_edges-1, 4*num_edges, dtype=self.begin.dtype, device=device)
            for i in range(num_edges-1):
                si = 4*i # start index
                fill = torch.tensor([0.0, 1.0, 2.0*t[i], 3.0*t[i]**2], dtype=self.begin.dtype)
                first[i, si:(si+4)] = fill
                first[i, (si+4):(si+8)] = -fill

            second = torch.zeros(num_edges-1, 4*num_edges, dtype=self.begin.dtype, device=device)
            for i in range(num_edges-1):
                si = 4*i # start index
                fill = torch.tensor([0.0, 0.0, 6.0*t[i], 2.0], dtype=self.begin.dtype)
                second[i, si:(si+4)] = fill
                second[i, (si+4):(si+8)] = -fill

            constraints = torch.cat((end_points, zeroth, first, second))
            self.constraints = constraints

            ## Compute null space, which forms our basis
            _, S, V = torch.svd(constraints, some=False)
            basis = V[:, S.numel():] # (num_coeffs)x(intr_dim)

            return basis

    def __init__(self, begin, end, num_nodes=5, basis=None, device=None, requires_grad=True):
        self.device = begin.device if device is None else device
        #begin # D or 1xD or BxD
        if len(begin) is 1 or begin.shape[0] is 1:
            self.begin = begin.detach().reshape((1, -1)) # 1xD
        else:
            self.begin = begin.detach() # BxD
        if len(end) is 1 or end.shape[0] is 1:
            self.end = end.detach().reshape((1, -1)) # 1xD
        else:
            self.end = end.detach() # BxD
        self.num_nodes = num_nodes
        if basis is None:
            self.basis = self.compute_basis(num_edges=num_nodes-1, device=self.device) # (num_coeffs)x(intr_dim)
        else:
            self.basis = basis
        self.parameters = torch.zeros(self.begin.shape[0], self.basis.shape[1], self.begin.shape[1],
                                      dtype=self.begin.dtype, device=self.device,
                                      requires_grad=requires_grad) # Bx(intr_dim)xD

    def __ppeval__(self, t, coeffs):
        # each row of coeffs should be of the form c0, c1, c2, ... representing polynomials
        # of the form c0 + c1*t + c2*t^2 + ...
        # coeffs: Bx(num_edges)x(degree)xD
        B, num_edges, degree, D = coeffs.shape
        idx = torch.floor(t.flatten() * num_edges).clamp(min=0, max=num_edges-1).long() # |t| # use this if nodes are equi-distant
        tpow = t.reshape((-1, 1)).pow(torch.arange(0.0, degree, dtype=t.dtype, device=self.device).reshape((1, -1))) # |t|x(degree)
        retval = torch.sum(tpow.unsqueeze(-1).expand(-1, -1, D).unsqueeze(0) * coeffs[:, idx], dim=2) # Bx|t|xD
        return retval

    def get_coeffs(self):
        coeffs = self.basis.unsqueeze(0).expand(self.parameters.shape[0], -1, -1).bmm(self.parameters) # Bx(num_coeffs)xD
        B, num_coeffs, D = coeffs.shape
        degree = 4
        num_edges = num_coeffs//degree
        coeffs = coeffs.reshape(B, num_edges, degree, D) # (num_edges)x4xD
        return coeffs

    def __call__(self, t):
        """Return the derivative of the curve at a given time point.
        """
        coeffs = self.get_coeffs() # Bx(num_edges)x4xD
        retval = self.__ppeval__(t, coeffs) # Bx|t|xD
        tt = t.reshape((-1, 1)).unsqueeze(0).expand(retval.shape[0], -1, -1) # Bx|t|x1
        retval += (1-tt).bmm(self.begin.unsqueeze(1)) + tt.bmm(self.end.unsqueeze(1)) # Bx|t|xD
        if retval.shape[0] is 1: # drop batching if we only have one element in the batch. XXX: This should probably be dropped in the future!
            retval.squeeze_(0) # |t|xD
        return retval

    def deriv(self, t):
        coeffs = self.get_coeffs() # Bx(num_edges)x4xD
        B, num_edges, degree, D = coeffs.shape
        dcoeffs = coeffs[:, :, 1:, :] * torch.arange(1.0, degree, dtype=t.dtype).reshape(1, 1, -1, 1).expand(B, num_edges, -1, D) # Bx(num_edges)x3xD
        retval = self.__ppeval__(t, dcoeffs) # Bx|t|xD
        #tt = t.reshape((-1, 1)) # |t|x1
        delta = (self.end - self.begin).unsqueeze(1) # Bx1xD
        retval += delta
        #if B is 1:
        #    retval.squeeze_(0) # drop batching if we only have one element in the batch. XXX: This should probably be dropped in the future!
        return retval

        # d + c*t + b*t^2 + a*t^3   =>
        # c + 2*b*t + 3*a*t^2

    def fit(self, t, x):
        """Fit the curve to the points by minimizing |x - c(t)|²

        Inputs:
            t:  a torch tensor with N elements showing where to evaluate the curve.
            x:  a torch tensor of size Nx(dim) containing the requested
                values the curve should take at time t.
        """
        optimizer = torch.optim.Adam([self.parameters], lr=1e-1)
        loss = torch.nn.MSELoss()
        for _ in range(1000):
            optimizer.zero_grad()
            L = loss(self(t), x)
            L.backward()
            optimizer.step()
            if self.parameters.grad.norm() < 1e-4:
                break
