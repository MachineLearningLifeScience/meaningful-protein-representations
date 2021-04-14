#!/usr/bin/env python3
import torch
import numpy as np
from .curve import *

def geodesic_minimizing_energy(curve, manifold, optimizer=torch.optim.Adam,
                               max_iter=150, eval_grid=20):
    """
    Compute a geodesic curve connecting two points by minimizing its energy.

    Mandatory inputs:
        curve:      A curve object representing a curve with fixed end-points.
                    When the function returns, this object has been updated to
                    be a geodesic curve.
        manifold:   A manifold object representing the space over which the
                    geodesic is defined. This object must provide a
                    'curve_energy' function through which pytorch can
                    back-propagate.

    Optional inputs:
        optimizer:  Choice of iterative optimizer.
                    Default: torch.optim.LBFGS
        max_iter:   The maximum number of iterations of the optimizer.
                    Default: 150
        eval_grid:  The number of points along the curve where
                    energy is evaluated.
                    Default: 20

    Output:
        success:    True if the algorithm converged, False otherwise.

    Example usage:
    S = Sphere()
    p0 = torch.tensor([0.1, 0.1]).reshape((1, -1))
    p1 = torch.tensor([0.3, 0.7]).reshape((1, -1))
    C = CubicSpline(begin=p0, end=p1, num_nodes=8, requires_grad=True)
    geodesic_minimizing_energy(C, S)
    """
    ## Initialize optimizer and set up closure
    alpha = torch.linspace(0, 1, eval_grid, device=curve.device).reshape((-1, 1))
    opt = optimizer([curve.parameters], lr=1e-1)
    def closure():
        opt.zero_grad()
        loss = manifold.curve_energy(curve(alpha)).mean()
        loss.backward()
        return loss

    thresh = 1e-4 # 1e-4
    for _ in range(max_iter):
        opt.step(closure=closure)
        if torch.max(torch.abs(curve.parameters.grad)) < thresh:
            break

    max_grad = torch.max(torch.abs(curve.parameters.grad))
    return max_grad < thresh


def geodesic_minimizing_energy_sgd(curve, manifold, optimizer=torch.optim.Adam,
                               max_iter=150, eval_grid=10, dt=0.005):
    """
    Compute a geodesic curve connecting two points by minimizing its energy.

    Mandatory inputs:
        curve:      A curve object representing a curve with fixed end-points.
                    When the function returns, this object has been updated to
                    be a geodesic curve.
        manifold:   A manifold object representing the space over which the
                    geodesic is defined. This object must provide a
                    'curve_energy' function through which pytorch can
                    back-propagate.

    Optional inputs:
        optimizer:  Choice of iterative optimizer.
                    Default: torch.optim.LBFGS
        max_iter:   The maximum number of iterations of the optimizer.
                    Default: 150
        eval_grid:  The number of points along the curve where
                    energy is evaluated.
                    Default: 20

    Output:
        success:    True if the algorithm converged, False otherwise.

    Example usage:
    S = Sphere()
    p0 = torch.tensor([0.1, 0.1]).reshape((1, -1))
    p1 = torch.tensor([0.3, 0.7]).reshape((1, -1))
    C = CubicSpline(begin=p0, end=p1, num_nodes=8, requires_grad=True)
    geodesic_minimizing_energy(C, S)
    """
    ## Initialize optimizer and set up closure
    opt = optimizer([curve.parameters], lr=1e-1)

    thresh = 5e-4 # 1e-4
    alpha_step = eval_grid*dt
    alpha_stop = 1.0 - alpha_step
    good_iters_in_a_row = 0
    for _ in range(max_iter):
        a = alpha_stop * torch.rand(1).item() # in [0, alpha_stop]
        alpha = torch.linspace(a, a+alpha_step, eval_grid, device=curve.device).reshape((-1, 1))
        def closure():
            opt.zero_grad()
            loss = manifold.curve_energy(curve(alpha)).mean()
            loss.backward()
            return loss
        opt.step(closure=closure)
        # XXX: determine a good convergence check
        if torch.max(torch.abs(curve.parameters.grad)) < thresh:
            good_iters_in_a_row += 1
        else:
            good_iters_in_a_row = 0
        if good_iters_in_a_row > 5:
            break

    max_grad = torch.max(torch.abs(curve.parameters.grad))
    return max_grad < thresh


class GeodesicODE(torch.nn.Module):
    """
    Wrapper class to evaluate the geodesic ODE of a given manifold
    as a first order ODE. The interface is compatible with 'torchdiffeq'.
    """
    def __init__(self, manifold):
        """
        Constructor for GeodesicODE.

        Input:
            manifold:   a manifold object representing the space
                        over which the geodesic is defined. This object
                        must provide a 'geodesic_system' function.
        """
        super(GeodesicODE, self).__init__()
        self.manifold = manifold

    def forward(self, t, x):
        """
        Evaluate the geodesic ODE as a first order ODE.

        Inputs:
            t:  a torch Tensor with T elements representing the time
                points where the ODE is evaluated.
            x:  a (2D)xT torch Tensor containing both points and
                derivatives long the solution curve. x[:D] contain
                T points of dimension D along the curve, while x[D:]
                contain first derivatives of the curve.

        Output:
            dx: a (2D)xT torch Tensor containing the first and second
                derivatives of the solution curve at the specified points.
                dx[:D] contain the first derivaties (identical to x[D:]),
                while dx[D:] contain second derivatives.
        """
        D = x.numel()//(t.numel()*2)
        c = x[:D].view(D, -1).t() # TxD
        dc = x[D:].view(D, -1).t() # TxD
        ddc = self.manifold.geodesic_system(c, dc)
        retval = torch.cat([dc, ddc], dim=1) # Tx(2D)
        return retval.t() # (2D)xT

    def f_numpy(self, t, x):
        """
        A numpy-based wrapper of the 'forward' function. The interface is
        identical to 'forward' except all inputs and outputs are numpy arrays.
        """
        torch_val = self.forward(torch.from_numpy(t).to(torch.float32),
                                 torch.from_numpy(x).to(torch.float32))
        return torch_val.numpy()


def shooting_geodesic(manifold, p, v, t=torch.linspace(0, 1, 50), requires_grad=False):
    """
    Compute the geodesic with a given starting point and initial velocity.

    Mandatory inputs:
        manifold:       the manifold over which the geodesic will be computed.
                        This object should provide a 'geodesic_system' function,
                        and should be compatible with the GeodesicODE class.
        p:              a torch Tensor with D elements representing the initial
                        position on the manifold of the requested geodesic.
        v:              a torch Tensor with D elements representing the initial
                        velocity of the requested geodesic.

    Optional inputs:
        t:              a torch Tensor of time values where the requested geodesic
                        will be computed. This must at least contain two values
                        where the first must be 0.
                        Default: torch.linspace(0, 1, 50)
        requires_grad:  if True it is possible to backpropagate through this
                        function.
                        Default: False

    Output:
        c:              a torch Tensor of size TxD containing points along the
                        geodesic at the reequested times.
        dc:             a torch Tensor of size TxD containing the curve derivatives
                        at the requested times.
    """
    #if requires_grad:
    #    from torchdiffeq import odeint_adjoint as odeint
    #else:
    #    from torchdiffeq import odeint
    from torchdiffeq import odeint
    odefunc = GeodesicODE(manifold)
    y = torch.cat([p.reshape(-1, 1), v.reshape(-1, 1)], dim=0) # (2D)xN
    retval = odeint(odefunc, y, t, method='rk4') # Tx(2D)xN
    D = retval.shape[1]//2
    c = retval[:, :D, :] # TxDxN
    dc = retval[:, D:, :] # TxDxN
    return c, dc

def connecting_shooting_geodesic(manifold, p0, p1, v_init=None):
    """
    This function does not work and should not be used.
    """
    if v_init is None:
        v = p1-p0
    else:
        v = v_init
    all_v = []
    v.requires_grad = True
    opt = torch.optim.LBFGS([v]) #SGD([v], lr=0.5)
    t = torch.linspace(0, 1, 2)
    def closure():
        all_v.append(v.detach().numpy().copy())
        opt.zero_grad()
        c, _ = shooting_geodesic(manifold, p0, v, t=t, requires_grad=True)
        loss = (c[-1] - p1).pow(2).sum()
        print(loss)
        loss.backward()
        #import matplotlib.pyplot as plt
        #plt.plot(c[:, 0].detach().numpy(), c[:, 1].detach().numpy())
        return loss

    for _ in range(50):
        opt.step(closure=closure)
        #print(torch.max(torch.abs(v.grad)))
        if torch.max(torch.abs(v.grad)) < 1e-2:
            break
    c, dc = shooting_geodesic(manifold, p0, v, t=t)
    return c, dc, v, all_v

def bvp_geodesic(manifold, p0, p1):
    """
    Compute the geodesic connecting two points by solving the associated
    boundary value problem using the SciPy 'solve_bvp' function.

    Inputs:
        manifold:   the manifold over which the geodesic will be computed.
                    This object should provide a 'geodesic_system' function,
                    and should be compatible with the GeodesicODE class.
        p0:         a torch Tensor representing the starting point of the
                    requested geodesic.
        p1:         a torch Tensor representing the ending point of the
                    requested geodesic.

    Output:
        retval:     an object returned by 'solve_bvp' representing the solution
                    curve.
    """
    odefunc_obj = GeodesicODE(manifold)
    odefunc = lambda t, x: odefunc_obj.f_numpy(t, x)
    p0np = p0.numpy().reshape(-1, 1)
    p1np = p1.numpy().reshape(-1, 1)
    dim = p0np.shape[0]

    def bc(ya, yb):
        retval = np.zeros(2*dim)
        retval[:dim] = ya[:dim] - p0np.flatten()
        retval[dim:] = yb[:dim] - p1np.flatten()
        return retval

    T = 30 # initial number of grid points
    t_init = np.linspace(0, 1, T, dtype=np.float32).reshape(1, T)
    line_init = np.outer(p0np, (1.0 - t_init)) + np.outer(p1np, t_init) # (dim)xT
    deriv_init = (p1np - p0np).reshape((dim, 1)).repeat(T, axis=1) # (dim)xT
    x_init = np.concatenate((line_init, deriv_init), axis=0) # (2dim)xT

    from scipy.integrate import solve_bvp
    retval = solve_bvp(odefunc, bc, t_init.flatten(), x_init)

    return retval
