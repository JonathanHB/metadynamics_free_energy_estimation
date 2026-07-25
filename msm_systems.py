import numpy as np


#system
class double_well:
    def __init__(self, A, B, D):
        self.A = A
        self.B = B
        self.D = D

    def G(self, x):
        return self.A * (x**4 - self.B*x**2)

    def x_minimum(self):
        return np.sqrt(self.B/2)
    
    def depth(self):
        return self.A*(self.B**2)/4
    
    def stdev(self):
        return np.sqrt(self.B/2)/2

    def well_volume(self,x):
        volume = 0
        for Gi in self.G(x):
            if Gi<0:
                volume -= Gi
        return volume
    
    def sec_deriv_bottom(self):
        #f'' = 12Ax**2 - 2AB
        return 4*self.A*self.B

    def bounds(self):
        neg_bound = -1.2*np.sqrt(self.B)
        return (neg_bound, -neg_bound)


#system
class cosine_well:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def G(self, x):
        return self.A * x**4 - self.B*np.cos(self.C*x)

    def x_minimum(self):
        return 0
    
    def depth(self):
        return 2*abs(self.B)
    
    def stdev(self):
        return 15
    
    def sec_deriv_bottom(self):
        #f'' = 12Ax**2 - 2AB
        return self.B*self.C**2

    def bounds(self):
        return (-10,10)