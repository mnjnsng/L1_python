import numpy as np
from scipy.linalg import null_space, inv, expm
from low_pass_filter import LTISystem
import matplotlib.pyplot as plt
import control

class L1_adapt(object):

    def __init__(self,f,g,wc=10,Ts=0.001):
        '''
        xdot = f(x) + g(x)u (control-affine structure)
        f: mapping from state space to state space (R^n)
        g: mapping from state space to R^(n x m) such that g(x)u makes R^n
        wc: cutoff frequency used in lowpass filter
        Ts: sampling time used in piece-wise continuous adaptation law
        '''
        #plant
        self.f = f
        self.g = g
        self.g_perp = lambda x : null_space(self.g(x).T)
        self.time = 0
    
        #low pass filter
        self.wc = wc # cutoff frequency
        self.Ts = Ts # sampling period
        self.lpf=LTISystem(A=np.array([-self.wc]),B=np.array([1]),C=np.array([self.wc]))       

        
        # Initialization of state, error and input vectors 
        self.x = np.zeros((4,1)) 
        self.x_tilde = np.zeros((4,1))
        self.u = np.zeros((1,1))
        self.n = self.g(self.x).shape[0]
        self.m = self.g(self.x).shape[1]


        # Initialize parameters needed for L1 controller
        self.As = -np.eye(self.n) # Choice of Hurwitz matrix used in piece-wise constant adaptation
        self.x_hat = np.zeros(shape = (self.n,1)) # Initialization of predicted state vector

    
    def update_error(self):
        return self.x_hat-self.x
    
    
    def plant(self,x,u):
        sigma_m = np.sin(0.5*self.time)
        #sigma_um = np.random.normal(0,0.1,size=(3,1))
        sigma_um = np.zeros((3,1))
        x_dot = self.f(x)+self.g(x)@(u+sigma_m) + self.g_perp(x)@(sigma_um)
        x_next = x + x_dot*self.Ts
 
        return x_next

    def adaptive_law(self,x_tilde):
        
        mat_expm = expm(self.As*self.Ts)
        Phi = inv(self.As) * (mat_expm - np.eye(self.n))
        adapt_gain = -inv(Phi)*mat_expm

        gg = np.concatenate((self.g(self.x),self.g_perp(self.x)),axis=1) #[g,g_perp]
        
        sigma_hat = inv(gg) @ adapt_gain @ x_tilde
        sigma_hat_m = sigma_hat[:self.m] 
        sigma_hat_um = sigma_hat[self.m:]

        return sigma_hat_m,sigma_hat_um

    
    def state_predictor(self,x,u,sigma_hat_m,sigma_hat_um):

        x_hat_dot = self.f(x)+self.g(x)@(u+sigma_hat_m)+np.matmul(self.g_perp(x),sigma_hat_um)+np.matmul(self.As,self.x_tilde)
        x_hat = self.x_hat + x_hat_dot*self.Ts #Euler extrapolation
        return x_hat

    def get_control_input(self,x,u_bl):

        sigma_hat_m, sigma_hat_um = self.adaptive_law(self.x_tilde)
        u_l1=-self.lpf.get_next_state(sigma_hat_m,self.Ts)
        u = u_bl+u_l1
        if u >= 1:
            u=np.array([[1]])
        elif u <=-1:
            u=np.array([[-1]])

        self.x = self.plant(x, u)
        self.x_hat = self.state_predictor(x,u,sigma_hat_m, sigma_hat_um)
        self.x_tilde = self.update_error()

        return u

