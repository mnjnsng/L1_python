import numpy as np
from scipy.linalg import null_space, inv, expm
from low_pass_filter import LTISystem
import matplotlib.pyplot as plt
import control


class L1_adapt(object):

    def __init__(self, env, f, g,x, wc=10, Ts=0.02):
        '''
        xdot = f(x) + g(x)u (control-affine structure)
        f: mapping from state space to state space (R^n)
        g: mapping from state space to R^(n x m) such that g(x)u makes R^n
        wc: cutoff frequency used in lowpass filter
        Ts: sampling time used in piece-wise continuous adaptation law
        '''
        # plant
        self._env = env
        self.f = f
        self.g = g
        self.g_perp = lambda x: null_space(self.g(x).T)
        self.time = 0

        # low pass filter
        self.wc = wc  # cutoff frequency
        self.Ts = Ts  # sampling period
        self.lpf = LTISystem(A=np.array(
            [-self.wc]), B=np.array([1]), C=np.array([self.wc]))

        # Initialization of state, error and input vectors
        # self.x = np.zeros((self._env.observation_space.shape[0], 1))
        self.x = x[...,np.newaxis]
        self.x_tilde = np.zeros((self._env.observation_space.shape[0], 1))
        self.u = np.zeros((self._env.action_space.shape[0], 1))
        self.n = self.g(self.x).shape[0]
        self.m = self.g(self.x).shape[1]
        self.force_mag = 10

        # Initialize parameters needed for L1 controller
        # Choice of Hurwitz matrix used in piece-wise constant adaptation
        self.As = -np.eye(self.n)
        # Initialization of predicted state vector
        self.x_hat = self.x

    def update_error(self):
        return self.x_hat-self.x

    def plant(self, x, u):
        
        
        self._env.env.steps_beyond_done = None
        self._env.env.state = np.array(x) 
        self.time += self.Ts
        return self._env.step(u.squeeze(0))[0] 

    def adaptive_law(self, x_tilde):

        mat_expm = expm(self.As*self.Ts)
        Phi = inv(self.As) * (mat_expm - np.eye(self.n))
        adapt_gain = -inv(Phi)*mat_expm

        gg = np.concatenate(
            (self.g(self.x), self.g_perp(self.x)), axis=1)  # [g,g_perp]

        sigma_hat = inv(gg) @ adapt_gain @ x_tilde
        sigma_hat_m = sigma_hat[:self.m]
        sigma_hat_um = sigma_hat[self.m:]

        return sigma_hat_m, sigma_hat_um

    def state_predictor(self, x, u, sigma_hat_m, sigma_hat_um):


        # print(self.f(x).shape)
        # print(self.g(x).shape)

        x_hat_dot = self.f(x)+self.g(x)@(u + sigma_hat_m)+np.matmul(
            self.g_perp(x), sigma_hat_um)+np.matmul(self.As, self.x_tilde)
        x_hat = self.x_hat + x_hat_dot*self.Ts  # Euler extrapolation
        return x_hat

    def get_control_input(self, x, u_bl):

        self.x = x[..., np.newaxis]
        self.x_tilde = self.update_error()
        sigma_hat_m, sigma_hat_um = self.adaptive_law(self.x_tilde)
        
        u_l1 = -self.lpf.get_next_state(sigma_hat_m)
        # u_l1 = -sigma_hat_m
        u = u_bl+u_l1

        if u >= self._env.action_space.high[0]:
            u = np.array([[self._env.action_space.high[0]]])
        elif u <= self._env.action_space.low[0]:
            u = np.array([[self._env.action_space.low[0]]])

        # self.x = self.plant(x, u)[..., np.newaxis]

        self.x_hat = self.state_predictor(
            x[..., np.newaxis], u, sigma_hat_m, sigma_hat_um)
        # self.x_tilde = self.update_error()

        
        return u.squeeze(0), sigma_hat_m.squeeze(0).squeeze(0), sigma_hat_um.squeeze(1), self.x_hat[2]
