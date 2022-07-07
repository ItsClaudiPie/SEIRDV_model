import numpy as np
from scipy.integrate import odeint

class RunModel:

    def __init__(self, model):
        self.model = model
    
    def run(self, days=720, **kwargs):
        t = np.linspace(0, days, days+1)
        x0 = self.model.init_run(**kwargs)
        x = odeint(self.model.get_derivatives, x0, t)
        
        if type(self.model.N) == int:
            x = [x[:, i] / x.sum(-1) for i in range(x.shape[1])]
        else:
            N = x.reshape(x.shape[0], self.model.n_districts, -1).sum(-1)
            N = N.repeat(x.shape[1] / N.shape[1], -1)
            x = [x[:, i] / N[:, i]
                for i in range(x.shape[1])]
        return t, x
