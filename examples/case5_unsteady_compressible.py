"""Case 5: CE-PINN for an unsteady compressible boundary layer.

This implementation retains the leading inner corrections ``ub0`` and ``hb0``
and the first normal-velocity correction ``vb1`` from ``ucbl_b1.py``. The
first-order outer acoustic fields are intentionally outside this example.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.main(x)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--npt", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reynolds", type=float, default=1e6)
    parser.add_argument("--output-dir", default="outputs/case5_unsteady_compressible")
    return parser.parse_args()


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.zeros_(module.bias)


def inner_reference_solution(xi, t, gamma=1.4, mach=100.0 / 340.0):
    eta = xi / torch.sqrt(t)
    erfc_term = torch.erfc(eta / 2.0)
    ub_ref = erfc_term
    hb_ref = (gamma - 1.0) * mach**2 * erfc_term * (1.0 - 0.5 * erfc_term)
    vb_ref = (
        (gamma - 1.0)
        * mach**2
        / torch.sqrt(torch.as_tensor(np.pi, dtype=xi.dtype, device=xi.device) * t)
        * (
            torch.erf(eta / np.sqrt(2.0)) / np.sqrt(2.0)
            - torch.exp(-(eta / 2.0) ** 2) * torch.erf(eta / 2.0)
            - 1.0 / np.sqrt(2.0)
        )
    )
    return ub_ref, hb_ref, vb_ref


def make_training_data(npt, batch_size, device, xi_end, t_start, t_end):
    xi = torch.linspace(0.0, xi_end, npt).view(-1, 1)
    t = torch.linspace(t_start, t_end, npt).view(-1, 1)
    xi_mesh, t_mesh = torch.meshgrid(xi.squeeze()[1:-1], t.squeeze()[1:], indexing="ij")
    dataset = TensorDataset(
        xi_mesh.reshape(-1, 1).to(device),
        t_mesh.reshape(-1, 1).to(device),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    t_bc = t.to(device)
    xi_wall = torch.zeros_like(t_bc)
    xi_far = xi_end * torch.ones_like(t_bc)
    xi_ic = xi.to(device)
    t_ic = t_start * torch.ones_like(xi_ic)
    return loader, xi_wall, xi_far, t_bc, xi_ic, t_ic


def physics_loss(net_ub0, net_hb0, net_vb1, xi, t, gamma, mach):
    xi = xi.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    inputs = torch.cat((xi, t), dim=1)

    ub0 = net_ub0(inputs)
    hb0 = net_hb0(inputs)
    vb1 = net_vb1(inputs)

    ub0_xi = grad(ub0, xi, torch.ones_like(ub0), create_graph=True)[0]
    ub0_t = grad(ub0, t, torch.ones_like(ub0), create_graph=True)[0]
    ub0_xixi = grad(ub0_xi, xi, torch.ones_like(ub0_xi), create_graph=True)[0]
    hb0_xi = grad(hb0, xi, torch.ones_like(hb0), create_graph=True)[0]
    hb0_t = grad(hb0, t, torch.ones_like(hb0), create_graph=True)[0]
    hb0_xixi = grad(hb0_xi, xi, torch.ones_like(hb0_xi), create_graph=True)[0]
    vb1_xi = grad(vb1, xi, torch.ones_like(vb1), create_graph=True)[0]

    mse = nn.MSELoss()
    return (
        mse(ub0_t - ub0_xixi, torch.zeros_like(ub0)),
        mse(
            hb0_t - hb0_xixi - (gamma - 1.0) * mach**2 * ub0_xi**2,
            torch.zeros_like(hb0),
        ),
        mse(vb1_xi - hb0_t, torch.zeros_like(vb1)),
    )


def boundary_loss(
    net_ub0,
    net_hb0,
    net_vb1,
    xi_wall,
    xi_far,
    t_bc,
    xi_ic,
    t_ic,
    gamma,
    mach,
):
    wall_inputs = torch.cat((xi_wall, t_bc), dim=1)
    far_inputs = torch.cat((xi_far, t_bc), dim=1)
    initial_inputs = torch.cat((xi_ic, t_ic), dim=1)

    xi_wall_grad = xi_wall.clone().detach().requires_grad_(True)
    t_wall_grad = t_bc.clone().detach().requires_grad_(True)
    hb0_wall = net_hb0(torch.cat((xi_wall_grad, t_wall_grad), dim=1))
    hb0_wall_xi = grad(
        hb0_wall,
        xi_wall_grad,
        torch.ones_like(hb0_wall),
        create_graph=True,
    )[0]

    ub_ref, hb_ref, vb_ref = inner_reference_solution(
        xi_ic, t_ic, gamma=gamma, mach=mach
    )
    mse = nn.MSELoss()
    return (
        mse(net_ub0(wall_inputs), torch.ones_like(xi_wall))
        + mse(net_ub0(far_inputs), torch.zeros_like(xi_far))
        + mse(hb0_wall_xi, torch.zeros_like(xi_wall))
        + mse(net_hb0(far_inputs), torch.zeros_like(xi_far))
        + mse(net_vb1(far_inputs), torch.zeros_like(xi_far))
        + mse(net_ub0(initial_inputs), ub_ref)
        + mse(net_hb0(initial_inputs), hb_ref)
        + mse(net_vb1(initial_inputs), vb_ref)
    )


def evaluate(net_ub0, net_hb0, net_vb1, device, xi_end, t_start, t_end, gamma, mach):
    xi = torch.linspace(0.0, xi_end, 101, device=device).view(-1, 1)
    t = torch.linspace(t_start, t_end, 41, device=device).view(-1, 1)
    xi_mesh, t_mesh = torch.meshgrid(xi.squeeze(), t.squeeze(), indexing="ij")
    xi_flat = xi_mesh.reshape(-1, 1)
    t_flat = t_mesh.reshape(-1, 1)
    inputs = torch.cat((xi_flat, t_flat), dim=1)

    with torch.no_grad():
        ub0 = net_ub0(inputs)
        hb0 = net_hb0(inputs)
        vb1 = net_vb1(inputs)
        ub_ref, hb_ref, vb_ref = inner_reference_solution(
            xi_flat, t_flat, gamma=gamma, mach=mach
        )
        errors = {
            "u": (torch.norm(ub0 - ub_ref) / torch.norm(ub_ref)).item(),
            "h": (
                torch.norm(hb0 - hb_ref) / (torch.norm(1.0 + hb_ref) + 1e-12)
            ).item(),
            "v": (torch.norm(vb1 - vb_ref) / (torch.norm(vb_ref) + 1e-12)).item(),
        }
    return xi_flat, t_flat, ub0, hb0, vb1, ub_ref, hb_ref, vb_ref, errors


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilon = 1.0 / args.reynolds
    gamma = 1.4
    mach = 100.0 / 340.0
    xi_end = 10.0
    t_start, t_end = 0.25, 1.0
    learning_rate = 1e-4

    net_ub0 = PINN().to(device)
    net_hb0 = PINN().to(device)
    net_vb1 = PINN().to(device)
    for network in (net_ub0, net_hb0, net_vb1):
        network.apply(initialize_weights)

    optimizer = torch.optim.Adam(
        list(net_ub0.parameters())
        + list(net_hb0.parameters())
        + list(net_vb1.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.99),
        eps=1e-15,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-6
    )
    loader, xi_wall, xi_far, t_bc, xi_ic, t_ic = make_training_data(
        args.npt, args.batch_size, device, xi_end, t_start, t_end
    )

    best_loss = float("inf")
    loss_history = []
    for epoch in range(1, args.epochs + 1):
        equation_total = 0.0
        boundary_total = 0.0
        batches = 0
        boundary_weight = 1.0 + 10.0 * epoch / args.epochs

        for xi_batch, t_batch in loader:
            optimizer.zero_grad()
            equation = sum(
                physics_loss(
                    net_ub0, net_hb0, net_vb1, xi_batch, t_batch, gamma, mach
                )
            )
            boundary = boundary_loss(
                net_ub0,
                net_hb0,
                net_vb1,
                xi_wall,
                xi_far,
                t_bc,
                xi_ic,
                t_ic,
                gamma,
                mach,
            )
            loss = equation + boundary_weight * boundary
            loss.backward()
            optimizer.step()
            equation_total += equation.detach()
            boundary_total += boundary.detach()
            batches += 1

        scheduler.step()
        loss_value = (
            equation_total / batches + boundary_weight * boundary_total / batches
        ).item()
        loss_history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save(
                {
                    "net_ub0": net_ub0.state_dict(),
                    "net_hb0": net_hb0.state_dict(),
                    "net_vb1": net_vb1.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss,
                    "epsilon": epsilon,
                    "gamma": gamma,
                    "mach": mach,
                },
                output_dir / "ckpt.pt",
            )

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:5d} | loss={loss_value:.6e} | best={best_loss:.6e}")

    checkpoint = torch.load(output_dir / "ckpt.pt", map_location=device)
    net_ub0.load_state_dict(checkpoint["net_ub0"])
    net_hb0.load_state_dict(checkpoint["net_hb0"])
    net_vb1.load_state_dict(checkpoint["net_vb1"])
    values = evaluate(
        net_ub0, net_hb0, net_vb1, device, xi_end, t_start, t_end, gamma, mach
    )
    xi, t, ub0, hb0, vb1, ub_ref, hb_ref, vb_ref, errors = values
    sqrt_epsilon = np.sqrt(epsilon)
    np.save(output_dir / "loss_hist.npy", np.asarray(loss_history))
    np.savez(
        output_dir / "result_data.npz",
        xi=xi.cpu().numpy(),
        t=t.cpu().numpy(),
        ub0=ub0.cpu().numpy(),
        hb0=hb0.cpu().numpy(),
        vb1=vb1.cpu().numpy(),
        u=ub0.cpu().numpy(),
        h=(1.0 + hb0).cpu().numpy(),
        rho=(1.0 / (1.0 + hb0)).cpu().numpy(),
        v=(sqrt_epsilon * vb1).cpu().numpy(),
        ub_ref=ub_ref.cpu().numpy(),
        hb_ref=hb_ref.cpu().numpy(),
        vb_ref=vb_ref.cpu().numpy(),
        epsilon=epsilon,
        **{f"l2_{name}": value for name, value in errors.items()},
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    train(parse_args())
