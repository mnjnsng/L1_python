import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import tqdm


class L1System:
    def __init__(self, Cf=51867, Cr=51867, Cf_nominal=56000, Cr_nominal=56000, V=30, R=20):
        # Modeled system
        """
        ẋ= Aₘ x + bₘ(ω uₐ + θ' x + σ)
        """
        self.km = torch.FloatTensor([[0.7223], [2.5855], [-0.6669], [0.1873]])
        self.A, self.b, self.g, self.p, _pbm = self.sys_matrices(V, R, Cf, Cr)
        A0, self.bm, self.g0, self.p, _pbm = self.sys_matrices(V, R, Cf_nominal, Cr_nominal)
        self.Am = A0 - self.bm * self.km.T

        self.ω = torch.FloatTensor([[Cf / Cf_nominal]])
        self.θ = torch.mm(1 / self.ω * _pbm, (self.A - self.Am)) + self.km.T * (1 / self.ω - 1)
        self.σ = torch.mm(_pbm, self.g) * (V / R)

        # State and hats Initialization
        self.x = torch.FloatTensor([[.1], [0], [0], [0]])
        self.x_hat = torch.FloatTensor([[.1], [0], [0], [0]])
        self.ω_hat = torch.FloatTensor([[0.1]])
        self.θ_hat = torch.FloatTensor([[0], [0], [0], [0]])
        self.σ_hat = torch.FloatTensor([[0]])
        self.η_hat = torch.FloatTensor([[0]])
        self.x_all = torch.cat((self.x, self.x_hat, self.ω_hat, self.θ_hat, self.σ_hat, self.η_hat), 0)

        # L1 parameters
        self.Gamma = 10000
        self.bandW = 90
        # P=lyap(Am',eye(4)); check ctrl pkg in python
        self.P = torch.FloatTensor([[1.9208, 0.0094, 0.2877, 0.0006],
                                    [0.0094, 0.0346, -0.0062, -0.0546],
                                    [0.2877, -0.0062, 10.2194, 0.0678],
                                    [0.0006, -0.0546, 0.0678, 0.0961]])

        # L1 solution
        self.X_sol_S = torch.tensor([])

    def l1_system(self, t, x_all):
        x = x_all[0:4]
        x_hat = x_all[4:8]
        ω_hat = x_all[9]
        θ_hat = x_all[9:13]
        σ_hat = x_all[13]
        η_hat = x_all[14]

        u_a = - η_hat
        x_tilde = x_hat - x

        # state predictor
        θ_hatx = torch.mm(θ_hat.T, x)
        u_all = ω_hat * u_a + θ_hatx + σ_hat
        x_hat_dot = torch.mm(self.Am, x_hat) + torch.mm(self.bm, u_all)

        # state
        x_dot = torch.mm(self.Am, x) + torch.mm(self.bm, u_all)

        # adaptive laws (+proj)
        xtP = torch.mm(x_tilde.T, self.P)
        xtPbm = torch.mm(xtP, self.bm)
        ω_hat_dot = - self.Gamma * xtPbm * u_a
        θ_hat_dot = - self.Gamma * xtPbm * x
        σ_hat_dot = - self.Gamma * xtPbm

        # control law
        η_hat_dot = - self.bandW * η_hat + self.bandW * (ω_hat * u_a + θ_hatx + σ_hat)

        return torch.cat((x_dot, x_hat_dot, ω_hat_dot, θ_hat_dot, σ_hat_dot, η_hat_dot), 0)

    def l1_control(self):
        t_interval = np.linspace(k, k + 1, t_samples + 1)
        x_sol = odeint(self.l1_system, self.x_all, torch.tensor(t_interval))
        self.x_all = x_sol[-1]
        self.X_sol_S = torch.cat([self.X_sol_S[0:-1], x_sol])

    def _feedback_system(self, t, x):
        u = self._u_feedback(x)
        u_all = self.ω * u + torch.mm(self.θ, x) + self.σ
        x_dot = torch.mm(self.Am, x) + torch.mm(self.bm, u_all)
        return x_dot

    def sim_nominal_system(self):
        t_interval = np.linspace(k, k + 1, t_samples + 1)
        x_sol = odeint(self._feedback_system, self.x, torch.tensor(t_interval))
        self.x = x_sol[-1]
        self.X_sol_S = torch.cat([self.X_sol_S[0:-1], x_sol])

    @staticmethod
    def _u_feedback(x):
        km = torch.FloatTensor([[0.7223], [2.5855], [-0.6669], [0.1873]])
        return - torch.mm(km.T, x)

    @staticmethod
    def sys_matrices(V, R, Cf, Cr, lf=1.1, lr=1.58, m=1573, Iz=2873):
        a1 = -2 * (Cf + Cr) / (m * V)
        a2 = 2 * (Cf + Cr) / m
        a3 = 2 * (-Cf * lf + Cr * lr) / (m * V)
        a4 = -2 * (Cf * lf - Cr * lr) / (Iz * V)
        a5 = 2 * (Cf * lf - Cr * lr) / Iz
        a6 = -2 * (Cf * lf ** 2 + Cr * lr ** 2) / (Iz * V)
        b1 = 2 * Cf / m
        b2 = 2 * Cf * lf / Iz
        g1 = -2 * (Cf * lf - Cr * lr) / (m * V) - V
        g2 = -2 * (Cf * lf ** 2 + Cr * lr ** 2) / (Iz * V)
        A = torch.FloatTensor([[0, 1, 0, 0], [0, a1, a2, a3], [0, 0, 0, 1], [0, a4, a5, a6]])
        b = torch.FloatTensor([[0], [b1], [0], [b2]])
        G = torch.FloatTensor([[0], [g1], [0], [g2]])
        p = torch.FloatTensor([[V / R]])

        _pbm = torch.FloatTensor([[0, m / Cf, 0, Iz / (Cf * lf)]]) / 4

        return A, b, G, p, _pbm


############################################################
############################################################
if __name__ == '__main__':

    time_step, t_samples = 100, 1
    sys = L1System()

    for k in tqdm.tqdm(range(time_step)):
        sys.l1_control()

    # plot #################
    t_space_all = np.linspace(0, time_step, t_samples * time_step + 1)
    plt.figure()
    plt.plot(t_space_all, sys.X_sol_S[:, 0], 'x-', linewidth=1, label='error')
    plt.title('error')
    plt.xlabel('t')
    plt.ylabel('x(1)')
    plt.legend()
    plt.show()
