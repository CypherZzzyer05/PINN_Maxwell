"""First-order Maxwell equations, no field labels or reference eigensolver."""

import torch
from torch import nn

from .geometry import Waveguide


def derivative(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


class LayerNet(nn.Module):
    def __init__(self, left, right, width):
        super().__init__()
        self.left, self.right = left, right
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, 2))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        # Start near a positive constant with a small magnetic field.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)

    def forward(self, x):
        t = 2 * (x - self.left) / (self.right - self.left) - 1
        raw = self.net(t)
        # A nodeless real mode selects the fundamental of this 1D TE problem.
        return torch.cat((torch.exp(raw[:, :1]), raw[:, 1:]), dim=1)


class MaxwellTEPINN(nn.Module):
    """Learn Ey=e and q=i Z0 Hz with e'=k0 q, q'=k0(neff²-n²)e.

    Exterior halfspaces use exact decaying exponentials. Positivity is specific
    to the fundamental TE mode, and must be removed for higher-order modes.
    """

    def __init__(self, geometry: Waveguide, width=32):
        super().__init__()
        self.geometry = geometry
        b = geometry.boundaries_um
        self.layers = nn.ModuleList(LayerNet(b[i], b[i + 1], width)
                                    for i in range(1, len(geometry.indices) - 1))
        self.raw_neff = nn.Parameter(torch.tensor(0.8))
        self.double()

    @property
    def neff(self):
        g = self.geometry
        return g.n_lower + (g.n_upper - g.n_lower) * torch.sigmoid(self.raw_neff)

    def point(self, x):
        return self.raw_neff.new_tensor([[x]])

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(Waveguide(**checkpoint["geometry"]), width=checkpoint["width"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()

    def loss_terms(self, samples):
        g = self.geometry
        pde = self.raw_neff.new_zeros(())
        for i, (net, points) in enumerate(zip(self.layers, samples), start=1):
            x = points.detach().requires_grad_(True)
            e, q = net(x).split(1, dim=1)
            r_e = derivative(e, x) / g.k0 - q
            r_q = derivative(q, x) / g.k0 - (self.neff**2 - g.indices[i]**2) * e
            pde = pde + r_e.square().mean() + r_q.square().mean()
        pde = pde / len(self.layers)
        interface = self.raw_neff.new_zeros(())
        for i in range(len(self.layers) - 1):
            x = self.point(g.boundaries_um[i + 2])
            interface = interface + (self.layers[i](x) - self.layers[i + 1](x)).square().mean()
        if len(self.layers) > 1:
            interface = interface / (len(self.layers) - 1)
        el, ql = self.layers[0](self.point(g.boundaries_um[1])).unbind(dim=1)
        er, qr = self.layers[-1](self.point(g.boundaries_um[-2])).unbind(dim=1)
        left_decay = torch.sqrt(self.neff**2 - g.indices[0]**2)
        right_decay = torch.sqrt(self.neff**2 - g.indices[-1]**2)
        boundary = ((ql - left_decay * el).square() +
                    (qr + right_decay * er).square()).mean()
        anchor_i = sum(g.anchor_um >= b for b in g.boundaries_um[2:-2])
        anchor = (self.layers[anchor_i](self.point(g.anchor_um))[0, 0] - 1).square()
        return {"pde": pde, "interface": interface, "boundary": boundary, "anchor": anchor}

    def objective(self, samples):
        t = self.loss_terms(samples)
        return t["pde"] + 10 * t["interface"] + 10 * t["boundary"] + 10 * t["anchor"]

    def sample(self, count, random=True):
        result = []
        for net in self.layers:
            if random:
                u = torch.rand(count, 1, dtype=self.raw_neff.dtype, device=self.raw_neff.device)
            else:
                u = (torch.arange(count, dtype=self.raw_neff.dtype,
                                  device=self.raw_neff.device).reshape(-1, 1) + 0.5) / count
            result.append(net.left + (net.right - net.left) * u)
        return result

    def forward(self, x):
        x = x.reshape(-1, 1)
        out = x.new_zeros((len(x), 2))
        g = self.geometry
        for net in self.layers:
            mask = ((x[:, 0] >= net.left) & (x[:, 0] < net.right))
            if mask.any():
                out[mask] = net(x[mask])
        for side in (0, 1):
            edge = g.boundaries_um[1] if side == 0 else g.boundaries_um[-2]
            mask = x[:, 0] < edge if side == 0 else x[:, 0] >= edge
            if mask.any():
                decay = torch.sqrt(self.neff**2 - g.indices[0 if side == 0 else -1]**2)
                sign = 1 if side == 0 else -1
                net = self.layers[0 if side == 0 else -1]
                e_edge = net(self.point(edge))[:, :1]
                e = e_edge * torch.exp(sign * g.k0 * decay * (x[mask] - edge))
                out[mask] = torch.cat((e, sign * decay * e), dim=1)
        return out
