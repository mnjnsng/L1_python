import numpy as np
from scipy.linalg import null_space, inv, expm
from scipy import signal

class L1_adapt:

    def __init__(self,f,g,x_current,u_bl,u_min= -np.infty, u_max = np.inf, wc=100,Ts=0.001):
        '''
        xdot = f(x) + g(x)u (control-affine structure)
        f: mapping from state space to state space (R^n)
        g: mapping from state space to R^(n x m) such that g(x)u makes R^n
        wc: cutoff frequency used in lowpass filter
        Ts: sampling time used in piece-wise continuous adaptation law
        '''
        self.f = f
        self.g = g
        self.g_perp = null_space(self.g(x_current).T)
        self.n = self.g(x_current).shape[0]
        self.m = self.g(x_current).shape[1]  # g is a n*m matrix
        self.wc = wc
        self.Ts = Ts
        self.x = x_current # Initialization of state vector 
        self.u_bl = u_bl
        self.u_min = u_min
        self.u_max = u_max


        # Initialize parameters needed for L1 controller
        self.As = -10*np.eye(self.n) # Choice of Hurwitz matrix used in piece-wise constant adaptation
        self.x = x_current
        self.x_hat = np.zeros(shape = (self.n,1)) # Initialization of predicted state vector
        self.x_tilde = np.zeros(shape = (self.n,1)) # Initialization of error
        self.gg = np.concatenate((self.g(x_current),self.g_perp),axis=1) #[g,g_perp]
        self.l = 0 # lowpass filter -equivalent state dynamics initial condition
        self.dt = 2*self.wc*self.Ts #low pass filter. Normalizing filtering frequency
        self.sigma_hat_m = np.zeros(shape = (self.m,1))
        self.sigma_hat_um = np.zeros(shape = (self.n-self.m,1) )

        self.t = 0

    
    def print_variables(self):
        print(f'f is {self.f}\n')
        print(f'g is {self.g}\n')
        print(f'g_perp is {self.g_perp}\n')
        print(f'n is {self.n}\n')
        print(f'm is {self.m}\n')
        print(f'x is {self.x}\n')
        print(f'u_bl is {self.u_bl}\n')
        print(f'As is {self.As}\n')
        print(f'gg is {self.gg}\n')
        print(f'l is {self.l}\n')
        print(f'dt is {self.f}\n')
        print(f'sigma_hat_m is {self.sigma_hat_m}\n')
        print(f'sigma_hat_um is {self.sigma_hat_um}\n')

    def update_error(self, x_hat,x):
        x_tilde = x_hat-x
        return x_tilde
    
    def plant(self,x,u):
        # This will be a gym environment.
        # this should update self.x
        x_dot = self.f(x)+self.g(x)*(u+0.5*np.sin(self.t)) # +np.matmul(self.g_perp,np.random.normal(0,0.1,size=(3,1)))
        x_new = x + x_dot*self.Ts
        self.t += self.Ts
        return x_new
        

    
    def adaptive_law(self, gg,x_tilde,As,Ts,n,m):
        
        mat_expm = expm(As*Ts)
        Phi = inv(As) * (mat_expm - np.eye(n))
        adapt_gain = -inv(Phi)*mat_expm
        sigma_hat = inv(gg) @ adapt_gain @ x_tilde
        sigma_hat_m = sigma_hat[:m] 
        sigma_hat_um = sigma_hat[m:]

        return sigma_hat_m,sigma_hat_um

    
    def state_predictor(self, f,g,g_perp, x,u,x_hat,sigma_hat_m,sigma_hat_um,As,Ts):
        
        x_hat_dot = f+g*(u+sigma_hat_m)+np.matmul(g_perp,sigma_hat_um)+np.matmul(As,(x_hat-x))
        x_hat = x + x_hat_dot*Ts #Euler extrapolation
        return x_hat

    
    def low_pass(self, sigma_hat_m, u_bl, wc, l, dt):
        
        a = -wc
        b = 1
        c = wc

        ldot = a*l + b*(sigma_hat_m-u_bl)
        l = l + dt*ldot
        u_l1 = c*l

        return u_l1, l
    

    def get_next_x_u(self,x,u):
        f = self.f(x)
        g = self.g(x)
        g_perp = null_space(g.T)
        gg = np.concatenate((g, g_perp),axis=1) #[g,g_perp]
        
        
        self.x_hat = self.state_predictor(f, g, g_perp, x, u, self.x_hat,self.sigma_hat_m, self.sigma_hat_um, self.As, self.Ts)
        self.x = self.plant(x, u)
        self.x_tilde = self.update_error(self.x_hat, self.x)
        self.sigma_hat_m, self.sigma_hat_um = self.adaptive_law(gg, self.x_tilde, self.As, self.Ts, self.n, self.m )
        
        u_bl = self.u_bl@self.x
        # u_l1, self.l = self.low_pass(self.sigma_hat_m, u_bl, self.wc, self.l, 0.001)
        sos = signal.butter(1, 100, 'lp', fs=self.dt, output='sos')
        u_l1 = -signal.sosfilt(sos, self.sigma_hat_m-u_bl)
        

        if u_l1 > self.u_max:
            u_next = self.u_max
        elif u_l1 < self.u_min:
            u_next = self.u_min
        else:
            u_next = u_l1

        return self.x, u_next

    def baseline_control(self, x,u):
        self.x = self.plant(x, u)
        u_bl = self.u_bl@self.x
        if u_bl > self.u_max:
            u_next = self.u_max
        elif u_bl < self.u_min:
            u_next = self.u_min
        else:
            u_next = u_bl
        return self.x, u_next

