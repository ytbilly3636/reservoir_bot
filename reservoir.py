import numpy as np
import cupy
import copy

# GPU
xp = cupy

class Reservoir(object):
    def __init__(self, i_size, r_size, i_coef=1.0, r_coef=0.999, sparse=0.5, leak=0.5):
        # i_size: size of input
        # r_size: size of reservoir
        # i_coef: intensity of weight connection between input and reservoir
        # r_coef: intensity of weight connection in reservoir
        # sparse: sparseness of weight connection in reservoir (sparse=1.0 -> no connection)
        # leak: leak rate (leak=0.0 -> no update)


        self.w_i = xp.random.uniform(-i_coef, i_coef, (r_size, i_size)).astype(xp.float32)
        
        self.w_r = np.random.uniform(-1, 1, (r_size * r_size, )).astype(np.float32)
        self.w_r[0:int(r_size * r_size * sparse)] = 0
        np.random.shuffle(self.w_r)
        self.w_r = self.w_r.reshape(r_size, r_size)
        self.w_r = self.w_r / max(abs(np.linalg.eig(self.w_r)[0])) * r_coef
        self.w_r = xp.array(self.w_r)
        
        self.leak = leak
        

    # reset reservoir state
    def reset(self, batch):
        # batch: batch size of u, an argument of __call__()
        
        self.x = xp.zeros((batch, self.w_r.shape[0]), dtype=xp.float32)
        

    # update reservoir state
    # return: updated reservoir state (batch, r_size)
    def __call__(self, u):
        # u: input (batch, i_size)

        if not u.shape[0] == self.x.shape[0]:
            print('different batchsize')
            print('required:', self.x.shape[0])
            print('actual:', u.shape[0])
        
        self.x = (1 - self.leak) * self.x + self.leak * xp.tanh(u.dot(self.w_i.T) + self.x.dot(self.w_r.T), dtype=xp.float32)
        
        return copy.deepcopy(self.x)
        
        
# ridge regression
# return: weight connection between reservoir and output (o_size, r_size)
def ridge_regression(x, t, norm=1.0):
    # x: reservoir states (T, r_size)
    # t: target signals (T, o_size)
    # norm: coef of regularization term

    array = x.T.dot(x)
    array = array + norm * xp.eye(x.shape[1])
    array = xp.linalg.inv(array)
    array = array.dot(x.T)
    array = array.dot(t)
    
    return array.T
