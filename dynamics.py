class dynamic_system:
    def __init__(self,t, x, u, ifMatched, ifUnmatched, dx, fx0, gx0, gxp):
        self.t = t
        self.x = x
        self.u = u
        self.ifMatched = ifMatched
        self.ifUnmatched = ifUnmatched
        self.dx = dx #unmatched uncertainty equation
        self.fx0 = fx0
        self.gx0 = gx0
        self.gxp = gxp
        
    def dynamics(self):
        #TODO: do what Yikun did in dynamics.m
        pass

    def get_fg(self):
        return self.fx0, self.gx0, self.gxp
        