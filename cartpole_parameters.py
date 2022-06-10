import numpy as np

class init_parameters():
    def __init__(self,dt, T, scale, nx):
        #simulation parameters
        self.dt = dt #simulation discrete timestep

        # L1 parameters
        self.scale = scale #sampling scale. Notice: control rate = dt/scale
        self.nx= 4 #number of state variables
        self.Ts = self.dt/self.scale #Control rate
        self.Ae = -10*np.eye(self.nx) 
        self.Mat_expm = np.exp(self.Ae*self.Ts)
        self.Phi = self.Ae / (self.Mat_expm- np.eye(nx))
        self.adapt_gain_no_Bm = -np.linalg.norm(self.Phi)*self.Mat_expm
        self.Kgain = 300 #rad/s


    def get_L1_parameters(self):
        return self.Ts, self.Kgain, self.adapt_gain_no_Bm
