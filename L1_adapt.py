import numpy as np
from scipy.linalg import null_space, inv, expm

class L1_adapt:

    def __init__(self,f,g,x_current,u_bl,wc=100,Ts=0.01):
        '''
        xdot = f(x) + g(x)u (control-affine structure)
        f: mapping from state space to state space (R^n)
        g: mapping from state space to R^(n x m) such that g(x)u makes R^n
        wc: cutoff frequency used in lowpass filter
        Ts: sampling time used in piece-wise continuous adaptation law
        '''
        self.f = f
        self.g = g
        self.g_perp = null_space(g)
        self.n = self.g.shape[0]
        self.m = self.g.shape[1]  # g is a n*m matrix
        self.wc = wc
        self.Ts = Ts
        self.x = x_current # Initialization of state vector 
        self.u_bl = u_bl


        # Initialize parameters needed for L1 controller
        self.As = -np.eye(self.n) # Choice of Hurwitz matrix used in piece-wise constant adaptation
        self.x_hat = np.zeros(self.n) # Initialization of predicted state vector
        self.x_tilde = np.zeros(self.n) # Initialization of error
        self.gg = np.concatenate((self.g.T,self.g_perp),axis=1) #[g,g_perp]
        self.l = 0 # lowpass filter -equivalent state dynamics initial condition
        self.dt = 2*self.wc*self.Ts #low pass filter. Normalizing filtering frequency
        self.sigma_hat_m = np.zeros(self.m)
        self.sigma_hat_um = np.zeros(self.n-self.m)


    def update_error(self):
        self.x_tilde = self.x_hat-self.x
            

    def plant(self,x,u):
        # This will be a gym environment.
        # this should update self.x
        pass

    def adaptive_law(self):
        
        mat_expm = expm(self.As*self.Ts)
        Phi = inv(self.As) * (mat_expm - np.eye(4))
        adapt_gain = -inv(Phi)*mat_expm
        sigma_hat = inv(self.gg) * adapt_gain * self.x_tilde
        self.sigma_hat_m = sigma_hat[:self.m] 
        self.sigma_hat_um = sigma_hat[self.m:]

    def state_predictor(self,x,u):
        x_hat_dot = self.f(x)+self.g(x)*(u+self.sigma_hat_m)+self.g_perp(x)*self.sigma_hat_um+self.As*self.x_tilde
        self.x_hat = self.x_hat + x_hat_dot*self.Ts #Euler extrapolation
        

    def low_pass(self):
        
        a = -self.wc
        b = 1
        c = self.wc

        ldot = a*self.l + b*self.sigma_hat_m
        self.l = self.l + self.dt*ldot
        u_l1 = c*self.l
    

    def get_next_u(self,x,u):

        self.predictor(x, u)
        self.plant(x, u)
        self.update_error()
        self.adaptive_law()
        u_l1 = self.low_pass()

        return self.u_bl+u_l1