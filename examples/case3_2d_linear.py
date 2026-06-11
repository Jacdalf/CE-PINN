import argparse
import numpy as np
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import TensorDataset, DataLoader

"""Case 3: CE-PINN for the 2D linear singularly perturbed elliptic problem.

ε(u_xx + u_yy) + u_x = -sin(πy)
Domain: x ∈ [0,1], y ∈ [0,1]
Boundary conditions: u = 0 on all boundaries
Running without ``--order`` trains orders 0, 1, and 2 sequentially.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--max-order", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--npt", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="outputs/case3_2d_linear")
    return parser.parse_args()


args = parse_args()
if args.order is None:
    for order in range(args.max_order + 1):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--order",
                str(order),
                "--epochs",
                str(args.epochs),
                "--npt",
                str(args.npt),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--output-dir",
                args.output_dir,
            ],
            check=True,
        )
    raise SystemExit(0)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#* a type of active function
class Swish(nn.Module):
		def __init__(self, inplace=True):
			super(Swish, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			if self.inplace:
				x.mul_(torch.sigmoid(x))
				return x
			else:
				return x * torch.sigmoid(x)

#* define a BL-PINN class
class Bl(nn.Module):
    def __init__(self, id, hd, od) -> None:
        super().__init__()
        self.main = nn.Sequential(nn.Linear(id, hd), 
                                  Swish(), 
                                  nn.Linear(hd, hd),
                                   Swish(),
                                  nn.Linear(hd, hd),
                                   Swish(),
                                  nn.Linear(hd, hd),
                                   Swish(),
                                #   nn.Linear(hd, hd),
                                #    Swish(), 
                                #   nn.Linear(hd, hd),
                                #    Swish(), 
                                #   nn.Linear(hd, hd),
                                #    Swish(), 
                                #   nn.Linear(hd, hd),
                                #    Swish(), 
                                  nn.Linear(hd, od))
    def forward(self, x):
        return self.main(x)

def freeze(*networks):
    for network in networks:
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)

def save_ckpt(savepath, **ckpt):
    torch.save(ckpt, savepath + 'ckpt.pt')

def save_loss(savepath, loss):
    np.save(savepath + 'loss_hist', loss)

def read_ckpt(filepath):
    return torch.load(filepath, map_location=device)

#* whether to use gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'using gpu: {torch.cuda.get_device_name()}')
else:
    print('using cpu')
    device = torch.device("cpu")

#* params
Re = 1e2
epsilon = 1. / Re
order = args.order
# lambda_bc = 1.
# lambda_v = 1.
# lambda_conti = 1.
lr = 1e-4
npt = args.npt
batch_size = args.batch_size
epochs = args.epochs
start_epoch = 1
test_interval = 20
log_interval = 5 * test_interval
loss_hist = []
best_loss = float('inf')  # 跟踪最小loss
rootpath = str(Path(args.output_dir).resolve()) + "/"
savepath = f"{rootpath}{order:d}/"
os.makedirs(savepath, exist_ok=True)

# geo
x_start = 0.
x_end = 1.
y_start = 0.
y_end = 1.
k_start = 0
k_end = 10
# df_y = k_end*epsilon**scale  # dividing factor of y

#* initialization
net_u_f0 = Bl(2, 64, 1)
net_u_f1 = Bl(2, 64, 1)
net_u_f2 = Bl(2, 64, 1)
net_u_g0 = Bl(2, 64, 1)
net_u_g1 = Bl(2, 64, 1)
net_u_g2 = Bl(2, 64, 1)
net_u_f0.to(device)
net_u_f1.to(device)
net_u_f2.to(device)
net_u_g0.to(device)
net_u_g1.to(device)
net_u_g2.to(device)
loss_fcn = nn.MSELoss(reduction='mean')
if order == 0:
    optimizer = torch.optim.Adam([
                                 {'params': net_u_f0.parameters()},
                                 {'params': net_u_g0.parameters()},
                                 ], lr, betas=(0.9, 0.99), eps=1e-15)
elif order == 1:
    ckpt = read_ckpt(rootpath + '/0/' + 'ckpt.pt')
    net_u_f0.load_state_dict(ckpt['net_u_f0'])
    net_u_g0.load_state_dict(ckpt['net_u_g0'])
    freeze(net_u_f0, net_u_g0)
    # Freeze parameters but allow gradient computation through the networks
    # for param in net_u_f0.parameters():
    #     param.requires_grad = False
    # for param in net_u_g0.parameters():
    #     param.requires_grad = False
    optimizer = torch.optim.Adam([
                                 {'params': net_u_f1.parameters()},
                                 {'params': net_u_g1.parameters()},
                                 ], lr, betas=(0.9, 0.99), eps=1e-15)
elif order == 2:
    ckpt = read_ckpt(rootpath + '/1/' + 'ckpt.pt')
    net_u_f0.load_state_dict(ckpt['net_u_f0'])
    net_u_g0.load_state_dict(ckpt['net_u_g0'])
    net_u_f1.load_state_dict(ckpt['net_u_f1'])
    net_u_g1.load_state_dict(ckpt['net_u_g1'])
    freeze(net_u_f0, net_u_g0, net_u_f1, net_u_g1)
    optimizer = torch.optim.Adam([
                                 {'params': net_u_f2.parameters()},
                                 {'params': net_u_g2.parameters()},
                                 ], lr, betas=(0.9, 0.99), eps=1e-15)
# optimizer = torch.optim.Adam([
#                              {'params': net_u_f0.parameters()},
#                              {'params': net_u_f1.parameters()},
#                              {'params': net_u_f2.parameters()},
#                              {'params': net_u_g0.parameters()},
#                              {'params': net_u_g1.parameters()},
#                              {'params': net_u_g2.parameters()},
#                             #  {'params': net_v_h0.parameters()},
#                             #  {'params': net_v_h1.parameters()},
#                             # #  {'params': net_v_l1.parameters()},
#                             #  {'params': net_p_q0.parameters()},
#                             #  {'params': net_p_q1.parameters()},
#                              ], lr, betas=(0.9, 0.99), eps=1e-15)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.7, cooldown=200)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, epochs), eta_min=1e-6
)

#* be used to get sample for training
def sample(npt, batch_size, device, x_start, x_end=1, y_start=0, y_end=1, k_start=0, k_end=10):
    # domain
    x = torch.linspace(x_start, x_end, npt).view(-1, 1)
    y = torch.linspace(y_start, y_end, npt).view(-1, 1)
    k = torch.linspace(k_start, k_end, npt).view(-1, 1)

    X, Y = torch.meshgrid(x.squeeze()[1:], y.squeeze()[1:], indexing='ij')
    xa = X.reshape(-1, 1).to(device)
    ya = Y.reshape(-1, 1).to(device)
    K, Y = torch.meshgrid(k.squeeze()[1:], y.squeeze()[1:], indexing='ij')
    ka = K.reshape(-1, 1).to(device)
    pt_set = TensorDataset(xa, ya, ka)
    pt_loader = DataLoader(pt_set, batch_size, shuffle=True)
    # pt_set = TensorDataset(x, k)
    # pt_loader = DataLoader(pt_set, batch_size, shuffle=True)

    # bc
    # xl = x_start * torch.ones(1, 1)
    # xr = x_end * torch.ones(1, 1)
    xl = x_start * torch.ones(npt, 1)
    xr = x_end * torch.ones(npt, 1)
    yb = y_start * torch.ones(npt, 1)
    yt = y_end * torch.ones(npt, 1)
    k0 = k_start * torch.ones(npt, 1)
    k1 = k_end * torch.ones(npt, 1)
    # kt = torch.tensor([k_end, 1.5*k_end, 2*k_end], dtype=torch.float32).view(-1, 1)
    # kt = torch.linspace(k_end, 2*k_end, int(npt/2)).view(-1, 1)
    return pt_loader, x.to(device), xl.to(device), xr.to(device), \
            y.to(device), yb.to(device), yt.to(device), \
            k.to(device), k0.to(device), k1.to(device)

def solve_eqn(order, xa, ya, ka):
    xa.requires_grad_(True)
    ya.requires_grad_(True)
    ka.requires_grad_(True)
    xy = torch.cat([xa, ya], dim=1)
    ky = torch.cat([ka, ya], dim=1)
    # xl = x_start * torch.ones_like(xa)
    # xl.requires_grad_(True)

    loss_eqn_outer = None

    if order == 0:
        f0 = net_u_f0(xy)
        df0_dx = grad(f0, xa, torch.ones_like(f0), create_graph=True)[0]
        loss_eqn_outer = loss_fcn(df0_dx + torch.sin(torch.pi*ya), torch.zeros_like(f0))
    elif order == 1:
        f0 = net_u_f0(xy)
        f1 = net_u_f1(xy)
        df0_dx = grad(f0, xa, torch.ones_like(f0), create_graph=True)[0]
        df0_dy = grad(f0, ya, torch.ones_like(f0), create_graph=True)[0]
        ddf0_dx2 = grad(df0_dx, xa, torch.ones_like(df0_dx), create_graph=True)[0]
        ddf0_dy2 = grad(df0_dy, ya, torch.ones_like(df0_dy), create_graph=True)[0]
        df1_dx = grad(f1, xa, torch.ones_like(f1), create_graph=True)[0]
        loss_eqn_outer = (0
                        #   + loss_fcn(df0_dx + f0, torch.zeros_like(f0))
                          + loss_fcn(ddf0_dx2 + ddf0_dy2 + df1_dx, torch.zeros_like(f1))
                          )
    elif order == 2:
        f1 = net_u_f1(xy)
        f2 = net_u_f2(xy)
        df1_dx = grad(f1, xa, torch.ones_like(f1), create_graph=True)[0]
        df1_dy = grad(f1, ya, torch.ones_like(f1), create_graph=True)[0]
        ddf1_dx2 = grad(df1_dx, xa, torch.ones_like(df1_dx), create_graph=True)[0]
        ddf1_dy2 = grad(df1_dy, ya, torch.ones_like(df1_dy), create_graph=True)[0]
        df2_dx = grad(f2, xa, torch.ones_like(f2), create_graph=True)[0]
        loss_eqn_outer = loss_fcn(
            ddf1_dx2 + ddf1_dy2 + df2_dx,
            torch.zeros_like(f2),
        )

    loss_eqn_inner = None

    if order == 0:
        g0 = net_u_g0(ky)
        dg0_dk = grad(g0, ka, torch.ones_like(g0), create_graph=True)[0]
        ddg0_dk2 = grad(dg0_dk, ka, torch.ones_like(dg0_dk), create_graph=True)[0]
        loss_eqn_inner = loss_fcn(ddg0_dk2 + dg0_dk, torch.zeros_like(g0))
    elif order == 1:
        g0 = net_u_g0(ky)
        g1 = net_u_g1(ky)
        # dg0_dk = grad(g0, k, torch.ones_like(g0), create_graph=True)[0]
        # ddg0_dk2 = grad(dg0_dk, k, torch.ones_like(dg0_dk), create_graph=True)[0]
        dg1_dk = grad(g1, ka, torch.ones_like(g1), create_graph=True)[0]
        ddg1_dk2 = grad(dg1_dk, ka, torch.ones_like(dg1_dk), create_graph=True)[0]
        loss_eqn_inner = (0
                        #   + loss_fcn(ddg0_dk2 + dg0_dk, torch.zeros_like(g0))
                          + loss_fcn(ddg1_dk2 + dg1_dk, torch.zeros_like(g1))
                          )
    elif order == 2:
        g0 = net_u_g0(ky)
        g2 = net_u_g2(ky)
        dg0_dy = grad(g0, ya, torch.ones_like(g0), create_graph=True)[0]
        ddg0_dy2 = grad(dg0_dy, ya, torch.ones_like(dg0_dy), create_graph=True)[0]
        dg2_dk = grad(g2, ka, torch.ones_like(g2), create_graph=True)[0]
        ddg2_dk2 = grad(dg2_dk, ka, torch.ones_like(dg2_dk), create_graph=True)[0]

        x_left = torch.zeros_like(xa).requires_grad_(True)
        f0_left = net_u_f0(torch.cat([x_left, ya], dim=1))
        df0_dx_left = grad(
            f0_left, x_left, torch.ones_like(f0_left), create_graph=True
        )[0]
        loss_eqn_inner = loss_fcn(
            ddg2_dk2 + dg2_dk + ddg0_dy2 + df0_dx_left * ka,
            torch.zeros_like(g2),
        )

    loss_eqn = loss_eqn_outer + loss_eqn_inner
    return loss_eqn

def solve_bc(order, x, xl, xr, y, yb, yt, k, k0, k1, lambda_k):
    x.requires_grad_(True)
    y.requires_grad_(True)
    k.requires_grad_(True)
    xl.requires_grad_(True)
    xr.requires_grad_(True)
    yb.requires_grad_(True)
    yt.requires_grad_(True)
    k0.requires_grad_(True)
    k1.requires_grad_(True)
    x_bl = k*epsilon
    xy_btm = torch.cat([x, yb], dim=1)[1:, :]
    xy_top = torch.cat([x, yt], dim=1)[1:, :]
    xy_left = torch.cat([xl, y], dim=1)
    xy_right = torch.cat([xr, y], dim=1)
    ky_btm = torch.cat([k, yb], dim=1)
    ky_top = torch.cat([k, yt], dim=1)
    ky_0 = torch.cat([k0, y], dim=1)
    ky_1 = torch.cat([k1, y], dim=1)
    xy_bl_top = torch.cat([x_bl, yt], dim=1)
    xy_bl_btm = torch.cat([x_bl, yb], dim=1)

    if order == 0:
       f0_left = net_u_f0(xy_left)
       f0_right = net_u_f0(xy_right)
       f0_btm = net_u_f0(xy_btm)
       f0_top = net_u_f0(xy_top)
       f0_bl_top = net_u_f0(xy_bl_top)
       f0_bl_btm = net_u_f0(xy_bl_btm)
       g0_0 = net_u_g0(ky_0)
       g0_1 = net_u_g0(ky_1)
       g0_top = net_u_g0(ky_top)
       g0_btm = net_u_g0(ky_btm)

       loss_left = loss_fcn(f0_left + g0_0, torch.zeros_like(f0_left))
       loss_right = loss_fcn(f0_right, torch.zeros_like(f0_right))
       loss_btm = loss_fcn(f0_btm, torch.zeros_like(f0_btm)) \
                    + loss_fcn(f0_bl_btm + g0_btm, torch.zeros_like(f0_bl_btm))
       loss_top = loss_fcn(f0_top, torch.zeros_like(f0_top)) \
                    + loss_fcn(f0_bl_top + g0_top, torch.zeros_like(f0_bl_top))
       loss_k1 = loss_fcn(g0_1, torch.zeros_like(g0_1))
    elif order == 1:
       f1_left = net_u_f1(xy_left)
       f1_right = net_u_f1(xy_right)
       f1_btm = net_u_f1(xy_btm)
       f1_top = net_u_f1(xy_top)
       f1_bl_top = net_u_f1(xy_bl_top)
       f1_bl_btm = net_u_f1(xy_bl_btm)
       g1_0 = net_u_g1(ky_0)
       g1_1 = net_u_g1(ky_1)
       g1_top = net_u_g1(ky_top)
       g1_btm = net_u_g1(ky_btm)

       loss_left = loss_fcn(f1_left + g1_0, torch.zeros_like(f1_left))
       loss_right = loss_fcn(f1_right, torch.zeros_like(f1_right))
       loss_btm = loss_fcn(f1_btm, torch.zeros_like(f1_btm)) \
                    + loss_fcn(f1_bl_btm + g1_btm, torch.zeros_like(f1_bl_btm))
       loss_top = loss_fcn(f1_top, torch.zeros_like(f1_top)) \
                    + loss_fcn(f1_bl_top + g1_top, torch.zeros_like(f1_bl_top))
       loss_k1 = loss_fcn(g1_1, torch.zeros_like(g1_1))
    elif order == 2:
       f2_left = net_u_f2(xy_left)
       f2_right = net_u_f2(xy_right)
       f2_btm = net_u_f2(xy_btm)
       f2_top = net_u_f2(xy_top)
       f2_bl_top = net_u_f2(xy_bl_top)
       f2_bl_btm = net_u_f2(xy_bl_btm)
       g2_0 = net_u_g2(ky_0)
       g2_1 = net_u_g2(ky_1)
       g2_top = net_u_g2(ky_top)
       g2_btm = net_u_g2(ky_btm)

       loss_left = loss_fcn(f2_left + g2_0, torch.zeros_like(f2_left))
       loss_right = loss_fcn(f2_right, torch.zeros_like(f2_right))
       loss_btm = loss_fcn(f2_btm, torch.zeros_like(f2_btm)) \
                    + loss_fcn(f2_bl_btm + g2_btm, torch.zeros_like(f2_bl_btm))
       loss_top = loss_fcn(f2_top, torch.zeros_like(f2_top)) \
                    + loss_fcn(f2_bl_top + g2_top, torch.zeros_like(f2_bl_top))
       loss_k1 = loss_fcn(g2_1, torch.zeros_like(g2_1))

    loss_bc = loss_left + loss_right + loss_btm + loss_top + lambda_k*loss_k1
    return loss_bc

def init_bl_weights(m):
    if isinstance(m, nn.Linear):
        # 对于主要的流场网络使用 Kaiming 初始化
        torch.nn.init.normal_(m.weight, mean=0., std=0.01)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

if order == 0:
    net_u_f0.apply(init_bl_weights)
    net_u_g0.apply(init_bl_weights)
elif order == 1:
    net_u_f1.apply(init_bl_weights)
    net_u_g1.apply(init_bl_weights)
else:
    net_u_f2.apply(init_bl_weights)
    net_u_g2.apply(init_bl_weights)

# 分别初始化不同的网络
# if order == 0:
#     net_u_f0.apply(init_bl_weights)
#     net_u_g0.apply(init_bl_weights)
# elif order == 1:
#     net_u_f1.apply(init_bl_weights)
#     net_u_g1.apply(init_bl_weights)
# elif order == 2:
#     net_u_f2.apply(init_bl_weights)
#     net_u_g2.apply(init_bl_weights)

#* train
# plt.ion()
# plt.figure()

# get sample
pt_loader, x, xl, xr, y, yb, yt, k, k0, k1 = sample(npt, batch_size, device, x_start, x_end, k_start, k_end)

# train
for epoch in range(start_epoch, epochs+1):
    # update lambda
    current_lambda_bc = 1.0 + 0. * epoch/epochs
    current_lambda_k = 1.0 + 0 * epoch/epochs
    # current_lambda_v = 1.0 + 1./epsilon * epoch/epochs
    # current_lambda_conti = 1.0 + 100. * epoch/epochs
    
    loss_eqn_total = 0
    loss_bc_total = 0
    count = 0
    
    for batch_id, (xa, ya, ka) in enumerate(pt_loader):
        optimizer.zero_grad()
        # cal loss
        # solve equation
        loss_eqn = solve_eqn(order, xa, ya, ka)
        # BC
        loss_bc = solve_bc(order, x, xl, xr, y, yb, yt, k, k0, k1, lambda_k=current_lambda_k)
        # total loss
        loss = 1.0 * loss_eqn + current_lambda_bc * loss_bc

        # backward grad and optimize
        loss.backward()
        optimizer.step()

        # loss accum
        count += 1
        loss_eqn_total += loss_eqn
        loss_bc_total += loss_bc
        # if batch_id % 30 == 0:
        #     print(f'''Train Epoch: {epoch}\teqn loss: {loss_eqn:.12f}\tbc_loss: {loss_bc:.12f}''')

    loss_eqn_total /= count
    loss_bc_total /= count
    loss_total = 1.0 * loss_eqn_total + current_lambda_bc * loss_bc_total
    # scheduler.step()
    # scheduler.step(loss_total)
    scheduler.step()

    loss_hist.append(loss_total.item())
    if loss_total.item() < best_loss:
        best_loss = loss_total.item()
        print(f"New best loss: {best_loss:.12f} at epoch {epoch}")
        save_ckpt(savepath,
                    order=order,
                    net_u_f0=net_u_f0.state_dict(),
                    net_u_f1=net_u_f1.state_dict(),
                    net_u_f2=net_u_f2.state_dict(),
                    net_u_g0=net_u_g0.state_dict(),
                    net_u_g1=net_u_g1.state_dict(),
                    net_u_g2=net_u_g2.state_dict(),
                #   net_v_h0=net_v_h0.state_dict(), # for fpbl_dbc.py, not used here. just for reference.
                #   net_v_h1=net_v_h1.state_dict(),
                #   net_v_l1=net_v_l1.state_dict(),
                #   net_p_q0=net_p_q0.state_dict(),
                #   net_p_q1=net_p_q1.state_dict(),
                    optim=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    epoch=epoch,
                    best_loss=best_loss)

    # test and log: save loss and ckeckpoint and record data, test and plot
    if epoch % test_interval == 0:
        if epoch % log_interval == 0:
            print(scheduler._last_lr)
            print(current_lambda_bc, current_lambda_k)
            save_loss(savepath, loss_hist)
        # print info
        print(f'''Train Epoch: {epoch}\tloss: {loss_total.data:.12f}\teqn loss: {loss_eqn_total.data:.12f}\tbc loss: {loss_bc_total.data:.12f}\tbest loss: {best_loss:.12f}''')
# plt.ioff()

# final save
save_loss(savepath, loss_hist)
print(f"Training completed. Best loss achieved: {best_loss:.12f}")
print(f"Best model checkpoint saved at: {savepath}ckpt.pt")


if __name__ == "__main__":
    pass

