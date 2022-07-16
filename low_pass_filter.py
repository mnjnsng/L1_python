import numpy as np

class LTISystem(object):
    def __init__(self,A,B,C,D=0.0):
        """Initialize the system."""
        self.x = np.zeros((1,1))
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def get_next_state(self,u,dt=0.001):
        state_change = self.A.dot(self.x) + self.B.dot(u)
        self.x += dt*state_change
        output = self.C.dot(self.x) + self.D*u
        return output
