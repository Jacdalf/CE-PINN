import argparse
import numpy as np
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import TensorDataset, DataLoader

"""Case 4: CE-PINN for the semi-infinite flat-plate boundary layer.

The retained terms correspond to the alternating outer/inner expansion used in
the paper through O(sqrt(epsilon)).
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--npt", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="outputs/case4_flat_plate")
    return parser.parse_args()


args = parse_args()
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

def save_ckpt(savepath, **ckpt):
    torch.save(ckpt, savepath + 'ckpt.pt')

def save_loss(savepath, loss):
    np.save(savepath + 'loss_hist', loss)

#* whether to use gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'using gpu: {torch.cuda.get_device_name()}')
else:
    print('using cpu')
    device = torch.device("cpu")

#* params
Re = 1e6
epsilon = 1. / Re
scale = 0.5
lr = 1e-4
npt = args.npt
batch_size = args.batch_size
epochs = args.epochs
start_epoch = 1
test_interval = 20
log_interval = 5 * test_interval
loss_hist = []
savepath = str(Path(args.output_dir).resolve()) + "/"
os.makedirs(savepath, exist_ok=True)
# geo
x_start = 0.25
x_end = 1.
y_start = 0.
y_end = 1.
k_start = 0
k_end = 10
# df_y = k_end*epsilon**scale  # dividing factor of y

#* initialization
net_u_f1 = Bl(2, 64, 1)
net_u_g0 = Bl(2, 64, 1)
net_v_h1 = Bl(2, 64, 1)
net_v_l1 = Bl(2, 64, 1)
net_p_q1 = Bl(2, 64, 1)
net_u_f1.to(device)
net_u_g0.to(device)
net_v_h1.to(device)
net_v_l1.to(device)
net_p_q1.to(device)
loss_fcn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam([
                             {'params': net_u_f1.parameters()},
                             {'params': net_u_g0.parameters()},
                             {'params': net_v_h1.parameters()},
                             {'params': net_v_l1.parameters()},
                             {'params': net_p_q1.parameters()},
                             ], lr, betas=(0.9, 0.99), eps=1e-15)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, epochs), eta_min=1e-6
)

#* be used to get sample for training
def sample(npt, batch_size, device, x_start, x_end=1,  y_start=0, y_end=1, k_start=0, k_end=10):
    # domain
    x = torch.linspace(x_start, x_end, npt).view(-1, 1)
    y = torch.linspace(y_start, y_end, npt).view(-1, 1)
    k = torch.linspace(k_start, k_end, npt).view(-1, 1)

    X, Y = torch.meshgrid(x.squeeze()[1:], y.squeeze()[1:], indexing='ij')
    xa = X.reshape(-1, 1).to(device)
    ya = Y.reshape(-1, 1).to(device)
    X, K = torch.meshgrid(x.squeeze()[1:], k.squeeze()[1:], indexing='ij')
    ka = K.reshape(-1, 1).to(device)
    pt_set = TensorDataset(xa, ya, ka)
    pt_loader = DataLoader(pt_set, batch_size, shuffle=True)
                        #    pin_memory=True, pin_memory_device=device)
    # bc
    xl = x_start * torch.ones(npt, 1)
    xr = x_end * torch.ones(npt, 1)
    yb = y_start * torch.ones(npt, 1)
    # yt = y_end * torch.ones(npt, 1)
    kt = k_end * torch.ones(npt, 1)
    return pt_loader, x.to(device), xl.to(device), xr.to(device), y.to(device), yb.to(device), k.to(device), kt.to(device)

def blasius_sol(y_coords, xpos, epsilon):
    # 四阶龙格库塔（RK4）
    eta_max = 10.0  # 固定eta的最大值
    npt = 1000     # 固定计算点数
    h = eta_max / npt  # 步长
    N = int(eta_max / h)

    # 初始化数组
    y1 = np.zeros(N) 
    y2 = np.zeros(N)
    y3 = np.zeros(N)
    y3[0] = 0.33205   # 初值

    # RK4求解
    for ii in range(N-1):
        k11 = y2[ii]
        k21 = y3[ii]
        k31 = -1/2 * y1[ii] * y3[ii]
        k12 = y2[ii] + h/2 * k21
        k22 = y3[ii] + h/2 * k31
        k32 = -1/2 * (y1[ii] + h/2 * k11) * (y3[ii] + h/2 * k31)
        k13 = y2[ii] + h/2 * k22
        k23 = y3[ii] + h/2 * k32
        k33 = -1/2 * (y1[ii] + h/2 * k12) * (y3[ii] + h/2 * k32)
        k14 = y2[ii] + h * k22
        k24 = y3[ii] + h * k32
        k34 = -1/2 * (y1[ii] + h * k13) * (y3[ii] + h * k33)
        y1[ii+1] = y1[ii] + h/6 * (k11 + 2*k12 + 2*k13 + k14)
        y2[ii+1] = y2[ii] + h/6 * (k21 + 2*k22 + 2*k23 + k24)
        y3[ii+1] = y3[ii] + h/6 * (k31 + 2*k32 + 2*k33 + k34)

    eta = np.arange(N) * h

    # 处理结果，当u接近1时设为1
    u = y2.copy()
    u[u > 0.999] = 1.0

    du_dx = -0.5*y3*eta/xpos
    du_dx[abs(du_dx) < 1e-6] = 0.0  # 当dudx足够小时设0

    v = 0.5*(epsilon/xpos)**0.5*(eta*y2-y1)
    dv_dx = -0.25*(epsilon**0.5)*xpos**-1.5*(y3*eta**2 + y2*eta - y1)

    # 对每个y坐标进行插值
    eta_physical = y_coords/(epsilon * xpos)**0.5
    u_interp = np.interp(eta_physical, eta, u)
    du_dx_interp = np.interp(eta_physical, eta, du_dx)
    v_interp = np.interp(eta_physical, eta, v)
    dv_dx_interp = np.interp(eta_physical, eta, dv_dx)
    # print(u_interp)
    # print(v_interp)
    # print(du_dx_interp)
    # print(dv_dx_interp)

    return u_interp, v_interp, du_dx_interp, dv_dx_interp

def solve_eqn(xa, ya, ka, lambda_conti):
    xa.requires_grad_(True)
    ya.requires_grad_(True)
    ka.requires_grad_(True)
    z = torch.zeros_like(xa)
    z.requires_grad_(True)
    xz = torch.cat((xa, z), 1)
    xy = torch.cat((xa, ya), 1)
    xk = torch.cat((xa, ka), 1)
    # f0 = net_u_f0(xy)
    f1 = net_u_f1(xy)
    # h0 = net_v_h0(xy)
    h1 = net_v_h1(xy)
    # q0 = net_p_q0(xy)
    q1 = net_p_q1(xy)
    # df0_dx = grad(f0, xa, torch.ones_like(f0), create_graph=True)[0]
    # df0_dy = grad(f0, ya, torch.ones_like(f0), create_graph=True)[0]
    df1_dx = grad(f1, xa, torch.ones_like(f1), create_graph=True)[0]
    # df1_dy = grad(f1, ya, torch.ones_like(f1), create_graph=True)[0]
    # dh0_dx = grad(h0, xa, torch.ones_like(h0), create_graph=True)[0]
    # dh0_dy = grad(h0, ya, torch.ones_like(h0), create_graph=True)[0]
    dh1_dx = grad(h1, xa, torch.ones_like(h1), create_graph=True)[0]
    dh1_dy = grad(h1, ya, torch.ones_like(h1), create_graph=True)[0]
    # dq0_dx = grad(q0, xa, torch.ones_like(q0), create_graph=True)[0]
    # dq0_dy = grad(q0, ya, torch.ones_like(q0), create_graph=True)[0]
    dq1_dx = grad(q1, xa, torch.ones_like(q1), create_graph=True)[0]
    dq1_dy = grad(q1, ya, torch.ones_like(q1), create_graph=True)[0]

    loss_conti_outer = lambda_conti * (
        0
        # loss_fcn(df0_dx + dh0_dy, torch.zeros_like(df0_dx))
        + loss_fcn(df1_dx + dh1_dy, torch.zeros_like(df1_dx))
    )
    loss_mmtx_outer = (
        0
        # loss_fcn(f0*df0_dx + h0*df0_dy + dq0_dx, torch.zeros_like(f0))
        # + loss_fcn(f0*df1_dx + h0*df1_dy + f1*df0_dx + h1*df0_dy + dq1_dx, torch.zeros_like(f1))
        + loss_fcn(df1_dx + dq1_dx, torch.zeros_like(f1))
    )
    loss_mmty_outer = (
        0
        # loss_fcn(f0*dh0_dx + h0*dh0_dy + dq0_dy, torch.zeros_like(q0))
        + loss_fcn(dh1_dx + dq1_dy, torch.zeros_like(q1))
    )
    loss_eqn_outer = loss_conti_outer + loss_mmtx_outer + loss_mmty_outer


    g0 = net_u_g0(xk)
    l1 = net_v_l1(xk)
    # f0_btm = net_u_f0(xz)
    # h0_btm = net_v_h0(xz)
    h1_btm = net_v_h1(xz)
    # df0_btm_dx = grad(f0_btm, xa, torch.ones_like(f0_btm), create_graph=True)[0]
    # dh0_btm_dz = grad(h0_btm, z, torch.ones_like(h0_btm), create_graph=True)[0]
    dg0_dx = grad(g0, xa, torch.ones_like(g0), create_graph=True)[0]
    dg0_dk = grad(g0, ka, torch.ones_like(g0), create_graph=True)[0]
    d2g0_dk2 = grad(dg0_dk, ka, torch.ones_like(dg0_dk), create_graph=True)[0]
    # dl1_dx = grad(l1, xa, torch.ones_like(l1), create_graph=True)[0]
    dl1_dk = grad(l1, ka, torch.ones_like(l1), create_graph=True)[0]

    loss_conti_inner = lambda_conti * (
        # 0
        loss_fcn(dg0_dx + dl1_dk, torch.zeros_like(g0))
    )
    loss_mmtx_inner = (
        0
        # + loss_fcn(g0*df0_btm_dx + (f0_btm + g0)*dg0_dx + (dh0_btm_dz + h1_btm + l1)*dg0_dk - d2g0_dk2, torch.zeros_like(g0)) # h1(x, 0) = 0
        + loss_fcn((1.0 + g0)*dg0_dx + (h1_btm + l1)*dg0_dk - d2g0_dk2, torch.zeros_like(g0)) # h1(x, 0) = 0
    )
    loss_mmty_inner = (
        0
    )
    loss_eqn_inner = loss_conti_inner + loss_mmtx_inner + loss_mmty_inner

    loss_eqn = loss_eqn_outer + loss_eqn_inner
    return loss_eqn

def solve_bc(x, xl, xr, y, yb, k, kt, f1_aly_xly, h1_aly_xly, u_bs_xly, v_bs_xly, f1_aly_xry, h1_aly_xry, u_bs_xry, v_bs_xry):
    xr_use = xr
    # xr_use.requires_grad_(True)
    xl_use = xl
    # xl_use.requires_grad_(True)
    ybl = k*epsilon**scale
    # y_bl.requires_grad_(True)
    # kt = kt*torch.sqrt(x)
    g0_xyb = net_u_g0(torch.cat((x, yb), 1))
    # h0_xyb = net_v_h0(torch.cat((x, yb), 1))
    h1_xyb = net_v_h1(torch.cat((x, yb), 1))
    l1_xyb = net_v_l1(torch.cat((x, yb), 1))

    g0_xkt = net_u_g0(torch.cat((x, kt), 1))
    l1_xkt = net_v_l1(torch.cat((x, kt), 1))

    # f0_xly = net_u_f0(torch.cat((xl_use, y), 1))
    f1_xly = net_u_f1(torch.cat((xl_use, y), 1))
    # h0_xly = net_v_h0(torch.cat((xl_use, y), 1))
    h1_xly = net_v_h1(torch.cat((xl_use, y), 1))
    # q0_xly = net_p_q0(torch.cat((xl_use, y), 1))
    # q1_xly = net_p_q1(torch.cat((xl_use, y), 1))
    f1_xry = net_u_f1(torch.cat((xr_use, y), 1))
    h1_xry = net_v_h1(torch.cat((xr_use, y), 1))


    # f0_xlybl = net_u_f0(torch.cat((xl_use, ybl), 1))
    h1_xlybl = net_v_h1(torch.cat((xl_use, ybl), 1))
    g0_xly = net_u_g0(torch.cat((xl_use, k), 1))
    l1_xly = net_v_l1(torch.cat((xl_use, k), 1))
    h1_xrybl = net_v_h1(torch.cat((xr_use, ybl), 1))
    g0_xry = net_u_g0(torch.cat((xr_use, k), 1))
    l1_xry = net_v_l1(torch.cat((xr_use, k), 1))

    loss_bottom = (
        0
        # lambda_v * loss_fcn(h0_xyb, torch.zeros_like(h0_xyb))
        # + loss_fcn(f0_xyb + g0_xyb, torch.zeros_like(f0_xyb))
        + loss_fcn(1.0 + g0_xyb, torch.zeros_like(g0_xyb))
        + loss_fcn(h1_xyb + l1_xyb, torch.zeros_like(h1_xyb))
    )

    loss_ktop = (
        loss_fcn(g0_xkt, torch.zeros_like(g0_xkt))
        + loss_fcn(l1_xkt, torch.zeros_like(l1_xkt))
    )

    loss_left = (
        0
        # + loss_fcn(f0_xly, torch.ones_like(f0_xly))
        # + loss_fcn(h0_xly, torch.zeros_like(h0_xly))
        # + loss_fcn(q0_xly, torch.zeros_like(q0_xly))
        + loss_fcn(f1_xly, f1_aly_xly)
        + loss_fcn(h1_xly, h1_aly_xly)
        # + loss_fcn(q1_xly, q1_aly_xly)
        # + loss_fcn(f0_xlybl + g0_xly, u_bs_xly)
        + loss_fcn(1.0 + g0_xly, u_bs_xly)
        + loss_fcn(h1_xlybl + l1_xly, v_bs_xly/epsilon**scale)
    )
    
    loss_right = (
        0
        + loss_fcn(f1_xry, f1_aly_xry)
        + loss_fcn(h1_xry, h1_aly_xry)
        # loss_fcn(f0_xry, torch.ones_like(f0_xry))
        # + loss_fcn(h0_xry, torch.zeros_like(h0_xry))
        # + loss_fcn(q0_xry, torch.zeros_like(q0_xry))
        + loss_fcn(1.0 + g0_xry, u_bs_xry)
        + loss_fcn(h1_xrybl + l1_xry, v_bs_xry/epsilon**scale)
    )


    loss_bc = loss_bottom + loss_ktop + loss_left + loss_right
    return loss_bc

def init_bl_weights(m):
    if isinstance(m, nn.Linear):
        # 对于主要的流场网络使用 Kaiming 初始化
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# 分别初始化不同的网络
net_u_f1.apply(init_bl_weights)
net_u_g0.apply(init_bl_weights)
net_v_h1.apply(init_bl_weights)
net_v_l1.apply(init_bl_weights)
net_p_q1.apply(init_bl_weights)

#* train
# plt.ion()
# plt.figure()

# get sample
pt_loader, x, xl, xr, y, yb, k, kt = sample(npt, batch_size, device, x_start, x_end, y_start, y_end, k_start, k_end)

# get analytical solution of psi1
ytemp = y.cpu().detach().numpy()
xltemp = xl.cpu().detach().numpy()
beta = 1.21678
fac1 = xltemp**2 + ytemp**2
fac2 = xltemp + np.sqrt(fac1)
# aly means analytical
f1_aly_xly = -beta*0.5*ytemp/(fac1**.5*fac2**.5)
h1_aly_xly = beta*0.5*(1 + xltemp/fac1**.5)/fac2**.5
f1_aly_xly = torch.tensor(f1_aly_xly, dtype=torch.float32).view(-1, 1).to(device)
h1_aly_xly = torch.tensor(h1_aly_xly, dtype=torch.float32).view(-1, 1).to(device)
# print(f1_aly_xly)
# print(h1_aly_xly)
xrtemp = xr.cpu().detach().numpy()
fac1 = xrtemp**2 + ytemp**2
fac2 = xrtemp + np.sqrt(fac1)
f1_aly_xry = -beta*0.5*ytemp/(fac1**.5*fac2**.5)
h1_aly_xry = beta*0.5*(1 + xrtemp/fac1**.5)/fac2**.5
f1_aly_xry = torch.tensor(f1_aly_xry, dtype=torch.float32).view(-1, 1).to(device)
h1_aly_xry = torch.tensor(h1_aly_xry, dtype=torch.float32).view(-1, 1).to(device)

# get blasius solution
ybl = k*epsilon**scale # ybl means y at boundary layer
u_bs_xly, v_bs_xly, _, _ = blasius_sol(ybl.cpu().detach().numpy(), x_start, epsilon)
u_bs_xry, v_bs_xry, _, _ = blasius_sol(ybl.cpu().detach().numpy(), x_end, epsilon)
u_bs_xly = torch.tensor(u_bs_xly, dtype=torch.float32).view(-1, 1).to(device)
v_bs_xly = torch.tensor(v_bs_xly, dtype=torch.float32).view(-1, 1).to(device)
u_bs_xry = torch.tensor(u_bs_xry, dtype=torch.float32).view(-1, 1).to(device)
v_bs_xry = torch.tensor(v_bs_xry, dtype=torch.float32).view(-1, 1).to(device)

# train
for epoch in range(start_epoch, epochs+1):
    # update lambda
    current_lambda_bc = 1.0 + 10. * epoch/epochs
    current_lambda_conti = 1.0 + 100. * epoch/epochs
    
    loss_eqn_total = 0
    loss_bc_total = 0
    count = 0
    
    for batch_id, (xa, ya, ka) in enumerate(pt_loader):
        optimizer.zero_grad()
        # cal loss
        # solve equation
        loss_eqn = solve_eqn(xa, ya, ka, current_lambda_conti)
        # BC
        loss_bc = solve_bc(x, xl, xr, y, yb, k, kt, f1_aly_xly, h1_aly_xly, u_bs_xly, v_bs_xly, f1_aly_xry, h1_aly_xry, u_bs_xry, v_bs_xry)
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

    # test and log: save loss and ckeckpoint and record data, test and plot
    if epoch % test_interval == 0:
        if epoch % log_interval == 0:
            print(scheduler._last_lr)
            print(current_lambda_bc, current_lambda_conti)
            save_loss(savepath, loss_hist)
            save_ckpt(savepath,
                      net_u_f1=net_u_f1.state_dict(),
                      net_u_g0=net_u_g0.state_dict(),
                      net_v_h1=net_v_h1.state_dict(),
                      net_v_l1=net_v_l1.state_dict(),
                      net_p_q1=net_p_q1.state_dict(),
                      optim=optimizer.state_dict(),
                      scheduler=scheduler.state_dict(),
                      epoch=epoch)
        # print info
        print(f'''Train Epoch: {epoch}\tloss: {loss_total.data:.12f}\teqn loss: {loss_eqn_total.data:.12f}\tbc loss: {loss_bc_total.data:.12f}''')
# plt.ioff()

# final save
# if not 'epoch' in locals():
    # epoch = epochs
save_loss(savepath, loss_hist)
save_ckpt(savepath,
          net_u_f1=net_u_f1.state_dict(),
          net_u_g0=net_u_g0.state_dict(),
          net_v_h1=net_v_h1.state_dict(),
          net_v_l1=net_v_l1.state_dict(),
          net_p_q1=net_p_q1.state_dict(),
          optim=optimizer.state_dict(),
          scheduler=scheduler.state_dict(),
          epoch=epoch)


if __name__ == "__main__":
    pass

