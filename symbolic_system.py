import pydrake.symbolic as sym
import numpy as np

def extract_variable_value_from_env(symbolic_var, env):
    # symbolic_var is a vector
    var_value = np.zeros(symbolic_var.shape[0])
    for i in range(symbolic_var.shape[0]):
        var_value[i] = env[symbolic_var[i]]
    return var_value

class LinearDynamics:
    def __init__(self, A, B, c):
        self.A = A
        self.B = B
        self.c = c

    def evaluate_xdot(self, x, u):
        return np.dot(self.A,x)+np.dot(self.B,u)+self.c

class ContinuousDynamics:
    def __init__(self, f, x, u):
        self.f = f
        self.x = x
        self.u = u
        self.h = sym.Variable("h")
        self._linearlize_system()

    def _linearlize_system(self):
        self.A = sym.Jacobian(self.f, self.x)
        self.B = sym.Jacobian(self.f, self.u)
        self.c = -(np.dot(self.A, self.x)+np.dot(self.B, self.u))+self.f

    def construct_linearized_system_at(self, env):
        return LinearDynamics(sym.Evaluate(self.A, env), sym.Evaluate(self.B, env), sym.Evaluate(self.c, env))

    def evaluate_xdot(self, env, linearize=False):
        if linearize:
            linsys = self.construct_linearized_system_at(env)
            x_env = extract_variable_value_from_env(self.x, env)
            u_env = extract_variable_value_from_env(self.u, env)
            return linsys.evaluate_xdot(x_env, u_env)
        else:
            return sym.Evaluate(sym.Evaluate(self.f, env))

class DTContinuousSystem:
    def __init__(self, f, x, u, initial_env=None):
        self.dynamics = ContinuousDynamics(f,x,u)
        if initial_env is None:
            self.env = {}
            for x_i in self.dynamics.x:
                self.env[x_i] = 0
        else:
            self.env = initial_env

    def foward_step(self, u=None, linearlize=False, modify_system=True, step_size = 1e-3, return_as_env = False):
        if not modify_system:
            new_env = self.env.copy()
        else:
            new_env = self.env
        if u is not None:
            for i in range(u.shape[0]):
                new_env[self.dynamics.u[i]] = u[i]
        else:
            for i in range(self.dynamics.u.shape[0]):
                new_env[self.dynamics.u[i]] = 0
        delta_x = self.dynamics.evaluate_xdot(new_env, linearlize)*step_size
        #assign new xs
        for i in range(delta_x.shape[0]):
            new_env[self.dynamics.x[i]] += delta_x[i]
        if return_as_env:
            return new_env
        else:
            return extract_variable_value_from_env(self.dynamics.x, new_env)

