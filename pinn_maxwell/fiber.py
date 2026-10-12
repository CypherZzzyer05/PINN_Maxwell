"""Full-vector m=1 dielectric fiber PINN with hard regularity and open boundary.

Time convention exp(-i omega t), propagation exp(+i beta z). Lengths: um.
State [u,w,p,q] means [Ez, -i Z0 Hz, E_phi, -i Z0 H_phi]
before applying the common phase exp(i phi + i beta z).
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import derivative


@dataclass(frozen=True)
class FiberGeometry:
    wavelength_um: float = 1.55
    interface_radii_um: tuple[float, ...] = (0.75, 1.5)
    indices: tuple[float, ...] = (1.6, 1.45, 1.44)
    radius_um: float = 4.0
    length_um: float = 10.0
    electric_amplitude_v_m: float = 1.0
    handedness: int = 1

    def __post_init__(self):
        values = (*self.interface_radii_um, *self.indices, self.wavelength_um,
                  self.radius_um, self.length_um, self.electric_amplitude_v_m)
        if not all(math.isfinite(t) for t in values):
            raise ValueError("All fiber parameters must be finite")
        if len(self.indices) < 2 or len(self.interface_radii_um) != len(self.indices)-1:
            raise ValueError("Need a core and at least one cladding; one radius per interface")
        if min(self.indices) <= 0 or self.wavelength_um <= 0:
            raise ValueError("Positive real indices and wavelength required")
        if self.interface_radii_um[0] <= 0 or any(b <= a for a, b in
                zip(self.interface_radii_um, self.interface_radii_um[1:])):
            raise ValueError("Interface radii must be positive and strictly increasing")
        if self.radius_um <= self.interface_radii_um[-1]:
            raise ValueError("Computational radius must extend into outer cladding")
        if self.indices[0] <= max(self.indices[1:]):
            raise ValueError("This core-guided branch requires the core index to be highest")
        if self.length_um <= 0 or self.electric_amplitude_v_m <= 0:
            raise ValueError("Positive propagation length and source amplitude required")
        if self.handedness not in (-1, 1):
            raise ValueError("handedness must be +1 or -1")

    @property
    def k0(self):
        return 2*math.pi/self.wavelength_um

    @property
    def n_lower(self):
        return max(self.indices[1:])

    def layer_at(self, radius):
        return np.searchsorted(self.interface_radii_um, radius, side="right")

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("interface_radii_um", "indices"):
            data[key] = tuple(data[key])
        return cls(**data)


class RadialNet(nn.Module):
    def __init__(self, left, right, width=32, core=False):
        super().__init__()
        self.left, self.right, self.core = left, right, core
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 4))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)

    def forward(self, radius):
        coordinate = radius.square()/self.right**2 if self.core else (radius-self.left)/(self.right-self.left)
        return self.net(2*coordinate-1)


class FiberPINN(nn.Module):
    """Learn a core-guided hybrid mode using four first-order Maxwell equations.

    Finite layers are neural fields. The infinite outer cladding is the unique
    decaying K1 solution, a differentiable exact Dirichlet-to-Neumann boundary.
    Analytic Bessel reference fields do not enter training.
    """

    def __init__(self, geometry: FiberGeometry, width=32):
        super().__init__()
        self.geometry = geometry
        boundaries = (0., *geometry.interface_radii_um)
        self.layers = nn.ModuleList(RadialNet(boundaries[i], boundaries[i+1], width, i == 0)
                                    for i in range(len(geometry.indices)-1))
        self.raw_neff = nn.Parameter(torch.tensor(0.0))
        self.raw_admittance = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        # K_n(x) = integral exp(-x cosh(t)) cosh(n t) dt. Scaled to avoid underflow.
        nodes, weights = np.polynomial.legendre.leggauss(96)
        self.register_buffer("cosh_nodes", torch.as_tensor(np.cosh(4*(nodes+1))))
        self.register_buffer("quad_weights", torch.as_tensor(4*weights))
        self.double()

    @property
    def neff(self):
        g = self.geometry
        return g.n_lower + (g.indices[0]-g.n_lower)*torch.sigmoid(self.raw_neff)

    def point(self, radius):
        return self.raw_neff.new_tensor([[radius]])

    def axis_values(self):
        # Exact Cartesian regularity at r=0 and unit transverse |E| on axis.
        p0 = self.raw_neff.new_tensor(1/math.sqrt(2))
        q0 = -p0*(0.5 + F.softplus(self.raw_admittance))
        eps = self.geometry.indices[0]**2
        u_slope = self.geometry.k0*(q0 + self.neff*p0)
        w_slope = self.geometry.k0*(self.neff*q0 + eps*p0)
        return u_slope, w_slope, p0, q0

    def finite_state(self, layer, radius):
        raw = self.layers[layer](radius)
        if layer:
            return raw
        u0, w0, p0, q0 = self.axis_values()
        correction = radius.square()*raw
        return torch.cat((radius*(u0+correction[:, :1]), radius*(w0+correction[:, 1:2]),
                          p0+correction[:, 2:3], q0+correction[:, 3:4]), dim=1)

    def scaled_k(self, argument):
        integrand = torch.exp(-argument*(self.cosh_nodes-1))*self.quad_weights
        return integrand.sum(dim=-1, keepdim=True), (integrand*self.cosh_nodes).sum(dim=-1, keepdim=True)

    def outer_state(self, radius):
        g = self.geometry
        edge = g.interface_radii_um[-1]
        edge_state = self.finite_state(len(self.layers)-1, self.point(edge))
        gamma = g.k0*torch.sqrt(self.neff**2-g.indices[-1]**2)
        k0_scaled, k1_scaled = self.scaled_k(gamma*radius)
        _, k1_edge = self.scaled_k(gamma*self.point(edge))
        ratio = torch.exp(-gamma*(radius-edge))*k1_scaled/k1_edge
        u, w = ratio*edge_state[:, :1], ratio*edge_state[:, 1:2]
        slope = -gamma*k0_scaled/k1_scaled-1/radius
        kt2 = -gamma**2
        beta = g.k0*self.neff
        p = (-beta*u/radius + g.k0*slope*w)/kt2
        q = (-beta*w/radius + g.k0*g.indices[-1]**2*slope*u)/kt2
        return torch.cat((u, w, p, q), dim=1)

    def forward(self, radius, layer=None):
        radius = radius.reshape(-1, 1)
        if layer is not None:
            return self.outer_state(radius) if layer == len(self.layers) else self.finite_state(layer, radius)
        result = radius.new_zeros((len(radius), 4))
        for i, net in enumerate(self.layers):
            mask = (radius[:, 0] >= net.left) & (radius[:, 0] < net.right)
            if mask.any():
                result[mask] = self.finite_state(i, radius[mask])
        outer = radius[:, 0] >= self.geometry.interface_radii_um[-1]
        if outer.any():
            result[outer] = self.outer_state(radius[outer])
        return result

    def sample(self, count, random=True):
        result = []
        for net in self.layers:
            if random:
                t = torch.rand(count, 1, dtype=torch.float64, device=self.raw_neff.device)
            else:
                t = (torch.arange(count, dtype=torch.float64, device=self.raw_neff.device).reshape(-1, 1)+0.5)/count
            # Keep singular-coordinate arithmetic away from the exact axis.
            result.append(net.left+(net.right-net.left)*(1e-4+(1-1e-4)*t))
        return result

    def loss_terms(self, samples):
        g = self.geometry
        pde = self.raw_neff.new_zeros(())
        for i, points in enumerate(samples):
            r = points.detach().requires_grad_(True)
            u, w, p, q = self.finite_state(i, r).split(1, dim=1)
            eps = g.indices[i]**2
            a = (self.neff*q-w/(g.k0*r))/eps  # Er=i a
            b = -self.neff*p+u/(g.k0*r)      # Z0 Hr=b
            residuals = (derivative(u, r)/g.k0 - (eps-self.neff**2)*q/eps - self.neff*w/(g.k0*eps*r),
                         derivative(w, r)/g.k0 - (eps-self.neff**2)*p - self.neff*u/(g.k0*r),
                         derivative(p, r)/g.k0 + (p+a)/(g.k0*r) + w,
                         derivative(q, r)/g.k0 + (q-b)/(g.k0*r) + eps*u)
            pde = pde + sum(t.square().mean() for t in residuals)/4
        pde = pde/len(self.layers)
        interface = self.raw_neff.new_zeros(())
        for i, radius in enumerate(g.interface_radii_um[:-1]):
            interface = interface + (self.finite_state(i, self.point(radius))-
                                      self.finite_state(i+1, self.point(radius))).square().mean()
        if len(self.layers) > 1:
            interface = interface/(len(self.layers)-1)
        edge = self.point(g.interface_radii_um[-1])
        boundary = (self.finite_state(len(self.layers)-1, edge)[:, 2:]-
                    self.outer_state(edge)[:, 2:]).square().mean()
        return {"pde": pde, "interface": interface, "boundary": boundary}

    def objective(self, samples):
        terms = self.loss_terms(samples)
        return terms["pde"]+10*terms["interface"]+10*terms["boundary"]

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        saved = torch.load(path, map_location=device, weights_only=True)
        model = cls(FiberGeometry(**saved["geometry"]), saved["width"]).to(device)
        model.load_state_dict(saved["state_dict"])
        return model.eval()
