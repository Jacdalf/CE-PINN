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

"""Case 2: CE-PINN for a 1D nonlinear convection-diffusion-reaction equation.

Running without ``--order`` trains orders 0, 1, and 2 sequentially.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--max-order", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--npt", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="outputs/case2_1d_nonlinear")
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
# y_start = 0.
# y_end = 1.
k_start = 0
k_end = 15
# normalization_scale = k_end/2
# df_y = k_end*epsilon**scale  # dividing factor of y

#* initialization
net_u_f0 = Bl(1, 64, 1)
net_u_f1 = Bl(1, 64, 1)
net_u_f2 = Bl(1, 64, 1)
net_u_g0 = Bl(1, 64, 1)
net_u_g1 = Bl(1, 64, 1)
net_u_g2 = Bl(1, 64, 1)
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
def sample_domain(npt, batch_size, device, x_start, x_end=1, k_start=0, k_end=10):
    # domain
    intervals_x = np.linspace(0, 1, npt + 1)[:-1]
    random_x = np.random.uniform(0, 1, size=npt) /npt
    lhs_x = intervals_x + random_x
    x = torch.tensor(lhs_x * (x_end - x_start) + x_start, dtype=torch.float32).view(-1, 1).to(device)
    # y = torch.linspace(y_start, y_end, npt).view(-1, 1)
    intervals_k = np.linspace(k_start, k_end, npt + 1)[:-1]
    random_k = np.random.uniform(0, 1, size=npt) / npt
    lhs_k = intervals_k + random_k
    k = torch.tensor(lhs_k * (k_end - k_start) + k_start, dtype=torch.float32).view(-1, 1).to(device)

    # X, Y = torch.meshgrid(x.squeeze()[1:], y.squeeze()[1:], indexing='ij')
    # xa = X.reshape(-1, 1).to(device)
    # ya = Y.reshape(-1, 1).to(device)
    # X, K = torch.meshgrid(x.squeeze()[1:], k.squeeze()[1:], indexing='ij')
    # ka = K.reshape(-1, 1).to(device)
    # pt_set = TensorDataset(xa, ya, ka)
    # pt_loader = DataLoader(pt_set, batch_size, shuffle=True)
    # dataset_x = TensorDataset(x)
    # dataset_k = TensorDataset(k)
    # loader_x = DataLoader(dataset_x, batch_size, shuffle=True)
    # loader_k = DataLoader(dataset_k, batch_size, shuffle=True)
    dataset_xk = TensorDataset(x, k)
    loader_xk = DataLoader(dataset_xk, batch_size, shuffle=True)

    # return loader_x, loader_k
    return loader_xk

def sample_bc(npt, device, x_start, x_end, k_start, k_end):
    xl = x_start * torch.ones(npt, 1)
    xr = x_end * torch.ones(npt, 1)
    k0 = k_start * torch.ones(npt, 1)
    k1 = k_end * torch.ones(npt, 1)
    return xl.to(device), xr.to(device), k0.to(device), k1.to(device)

def solve_eqn(order, x, k):
    x.requires_grad_(True)
    k.requires_grad_(True)
    xl = x_start * torch.ones_like(x)
    xl.requires_grad_(True)

    loss_eqn_outer = None

    if order == 0:
        f0 = net_u_f0(x)
        df0_dx = grad(f0, x, torch.ones_like(f0), create_graph=True)[0]
        loss_eqn_outer = loss_fcn(df0_dx - f0**2 + 1, torch.zeros_like(f0))
    elif order == 1:
        f0 = net_u_f0(x)
        f1 = net_u_f1(x)
        df0_dx = grad(f0, x, torch.ones_like(f0), create_graph=True)[0]
        ddf0_dx2 = grad(df0_dx, x, torch.ones_like(df0_dx), create_graph=True)[0]
        df1_dx = grad(f1, x, torch.ones_like(f1), create_graph=True)[0]
        loss_eqn_outer = (0
                          + loss_fcn(ddf0_dx2 + df1_dx - 2*f0*f1, torch.zeros_like(f1))
                          )
    elif order == 2:
        f0 = net_u_f0(x)
        f1 = net_u_f1(x)
        f2 = net_u_f2(x)
        df1_dx = grad(f1, x, torch.ones_like(f1), create_graph=True)[0]
        ddf1_dx2 = grad(df1_dx, x, torch.ones_like(df1_dx), create_graph=True)[0]
        df2_dx = grad(f2, x, torch.ones_like(f2), create_graph=True)[0]
        loss_eqn_outer = (0 
                          + loss_fcn(ddf1_dx2 + df2_dx - f1**2 - 2*f0*f2, torch.zeros_like(f2))
                          )

    loss_eqn_inner = None

    if order == 0:
        g0 = net_u_g0(k)
        dg0_dk = grad(g0, k, torch.ones_like(g0), create_graph=True)[0]
        # ddg0_dk2 = grad(dg0_dk, k, torch.ones_like(dg0_dk), create_graph=True)[0]
        # loss_eqn_inner = loss_fcn(ddg0_dk2 + dg0_dk, torch.zeros_like(g0))
        loss_eqn_inner = loss_fcn(dg0_dk + g0, torch.zeros_like(g0))
    elif order == 1:
        g0 = net_u_g0(k)
        g1 = net_u_g1(k)
        dg1_dk = grad(g1, k, torch.ones_like(g1), create_graph=True)[0]
        ddg1_dk2 = grad(dg1_dk, k, torch.ones_like(dg1_dk), create_graph=True)[0]
        f0_l = net_u_f0(xl)
        loss_eqn_inner = (0
                          + loss_fcn(ddg1_dk2 + dg1_dk - g0**2 - 2*f0_l*g0, torch.zeros_like(g1))
                          )
    elif order == 2:
        g0 = net_u_g0(k)
        g1 = net_u_g1(k)
        g2 = net_u_g2(k)
        dg2_dk = grad(g2, k, torch.ones_like(g2), create_graph=True)[0]
        ddg2_dk2 = grad(dg2_dk, k, torch.ones_like(dg2_dk), create_graph=True)[0]
        f0_l = net_u_f0(xl)
        df0_dx_l = grad(f0_l, xl, torch.ones_like(f0_l), create_graph=True)[0]
        ddf0_dx2_l = grad(df0_dx_l, xl, torch.ones_like(df0_dx_l), create_graph=True)[0]
        f1_l = net_u_f1(xl)
        loss_eqn_inner = (0 
                          + loss_fcn(ddg2_dk2 + dg2_dk + ddf0_dx2_l*k - 2*(f0_l + g0)*(df0_dx_l*k + g1) - 2*g0*f1_l, torch.zeros_like(g2))
                          )

    loss_eqn = loss_eqn_outer + 5*loss_eqn_inner
    return loss_eqn

def solve_bc(order, xl, xr, k0, k1, lambda_k):
    xl.requires_grad_(True)
    xr.requires_grad_(True)
    k0.requires_grad_(True)
    k1.requires_grad_(True)
    loss_left = None
    loss_right = None
    loss_ktop = None

    if order == 0:
        f0_xl = net_u_f0(xl)
        f0_xr = net_u_f0(xr)
        g0_k0 = net_u_g0(k0)
        # k1.requires_grad_(True)
        # g0_k1 = net_u_g0(k1)
        # dg0_dk_k1 = grad(g0_k1, k1, torch.ones_like(g0_k1), create_graph=True)[0]
        loss_left = loss_fcn(f0_xl + g0_k0, torch.zeros_like(f0_xl))
        loss_right = loss_fcn(f0_xr, torch.zeros_like(f0_xr))
        loss_ktop = 0 \
                    # + loss_fcn(g0_k1, torch.zeros_like(g0_k1)) \
                    # + loss_fcn(dg0_dk_k1, torch.zeros_like(dg0_dk_k1))
    elif order == 1:
        # f0_xl = net_u_f0(xl)
        f1_xl = net_u_f1(xl)
        # g0_K0 = net_u_g0(k0)
        g1_k0 = net_u_g1(k0)
        # f0_xr = net_u_f0(xr)
        f1_xr = net_u_f1(xr)
        # g0_xkt = net_u_g0(kt)
        g1_k1 = net_u_g1(k1)
        loss_left = (0
                    #  + loss_fcn(f0_xl + g0_xl, torch.zeros_like(f0_xl))
                     + loss_fcn(f1_xl + g1_k0, torch.zeros_like(f1_xl))
                     )
        loss_right = (0
                    #   + loss_fcn(f0_xr, torch.ones_like(f0_xr))
                      + loss_fcn(f1_xr, torch.zeros_like(f1_xr))
                      )
        loss_ktop = (0
                    #  + loss_fcn(g0_xkt, torch.zeros_like(g0_xkt))
                     + loss_fcn(g1_k1, torch.zeros_like(g1_k1))
                     )
    elif order == 2:
        # f0_xl = net_u_f0(xl)
        # f1_xl = net_u_f1(xl)
        f2_xl = net_u_f2(xl)
        # g0_xl = net_u_g0(xl)
        # g1_xl = net_u_g1(xl)
        g2_k0 = net_u_g2(k0)
        # f0_xr = net_u_f0(xr)
        # f1_xr = net_u_f1(xr)
        f2_xr = net_u_f2(xr)
        # g0_xkt = net_u_g0(kt)
        # g1_xkt = net_u_g1(kt)
        g2_k1 = net_u_g2(k1)
        loss_left = (0 
                    #  + loss_fcn(f0_xl + g0_xl, torch.zeros_like(f0_xl))
                    #  + loss_fcn(f1_xl + g1_xl, torch.zeros_like(f1_xl))
                     + loss_fcn(f2_xl + g2_k0, torch.zeros_like(f2_xl))
                     )
        loss_right = (0 
                    #   + loss_fcn(f0_xr, torch.ones_like(f0_xr))
                    #   + loss_fcn(f1_xr, torch.zeros_like(f1_xr))
                      + loss_fcn(f2_xr, torch.zeros_like(f2_xr))
                      )
        loss_ktop = (0 
                    #  + loss_fcn(g0_xkt, torch.zeros_like(g0_xkt))
                    #  + loss_fcn(g1_xkt, torch.zeros_like(g1_xkt))
                     + loss_fcn(g2_k1, torch.zeros_like(g2_k1))
                     )

    loss_bc = lambda_k*loss_ktop + loss_left + loss_right
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

#* train
# plt.ion()
# plt.figure()

# get sample
loader_xk = sample_domain(npt, batch_size, device, x_start, x_end, k_start, k_end)
xl, xr, k0, k1 = sample_bc(npt, device, x_start, x_end, k_start, k_end)

# train
for epoch in range(start_epoch, epochs+1):
    # update lambda
    current_lambda_bc = 1.0
    # current_lambda_bc = 1.0 + 10. * epoch/epochs
    # current_lambda_k = 1.0
    current_lambda_k = 1.0 + 10 * epoch/epochs
    # current_lambda_v = 1.0 + 1./epsilon * epoch/epochs
    # current_lambda_conti = 1.0 + 100. * epoch/epochs
    
    loss_eqn_total = 0
    loss_bc_total = 0
    count = 0

    if epoch % 100 == 0:
        loader_xk = sample_domain(npt, batch_size, device, x_start, x_end, k_start, k_end)
    
    for batch_id, (x, k) in enumerate(loader_xk):
        optimizer.zero_grad()
        # cal loss
        # solve equation
        loss_eqn = solve_eqn(order, x, k)
        # BC
        loss_bc = solve_bc(order, xl, xr, k0, k1, lambda_k=current_lambda_k)
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

