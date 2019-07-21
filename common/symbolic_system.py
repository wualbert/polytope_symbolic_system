import pydrake.symbolic as sym
import numpy as np
from pypolycontain.lib.zonotope import *

def extract_variable_value_from_env(symbolic_var, env):
    # symbolic_var is a vector
    var_value = np.zeros(symbolic_var.shape[0])
    for i in range(symbolic_var.shape[0]):
        var_value[i] = env[symbolic_var[i]]
    return var_value

class Dynamics:
    def __init__(self):
        self.type='undefined'
        pass

    def construct_linearized_system_at(self, env):
        raise NotImplementedError


class ContinuousLinearDynamics(Dynamics):
    def __init__(self, A, B, c):
        Dynamics.__init__(self)
        self.A = A
        self.B = B
        self.c = c
        self.type='continuous'

    def construct_linearized_system_at(self, env):
        print('Warning: system is already linear!')
        return self

    def evaluate_xdot(self, x, u):
        return np.dot(self.A,x)+np.dot(self.B,u)+self.c


class ContinuousDynamics(Dynamics):
    '''
    System described by xdot(t) = f(x(t), u(t))
    '''
    def __init__(self, f, x, u):
        Dynamics.__init__(self)
        self.f = f
        self.x = x
        self.u = u
        self.type='continuous'
        self._linearlize_system()

    def _linearlize_system(self):
        self.A = sym.Jacobian(self.f, self.x)
        self.B = sym.Jacobian(self.f, self.u)
        self.c = -(np.dot(self.A, self.x)+np.dot(self.B, self.u))+self.f

    def construct_linearized_system_at(self, env):
        return ContinuousLinearDynamics(sym.Evaluate(self.A, env), sym.Evaluate(self.B, env), sym.Evaluate(self.c, env))

    def evaluate_xdot(self, env, linearize=False):
        if linearize:
            linsys = self.construct_linearized_system_at(env)
            x_env = extract_variable_value_from_env(self.x, env)
            u_env = extract_variable_value_from_env(self.u, env)
            return linsys.evaluate_xdot(x_env, u_env)
        else:
            return sym.Evaluate(self.f, env)

class DiscreteLinearDynamics(Dynamics):
    def __init__(self, A, B, c):
        Dynamics.__init__(self)
        self.A = A
        self.B = B
        self.c = c
        self.type='discrete'

    def construct_linearized_system_at(self, env):
        print('Warning: system is already linear!')
        return self

    def evaluate_x_next(self, x, u):
        return np.dot(self.A,x)+np.dot(self.B,u)+self.c

class DiscreteDynamics(Dynamics):
    '''
    System described by x[t+1] = f(x[t], u[t])
    '''
    def __init__(self, f, x, u):
        Dynamics.__init__(self)
        self.f = f
        self.x = x
        self.u = u
        self.type='discrete'
        self._linearlize_system()

    def _linearlize_system(self):
        self.A = sym.Jacobian(self.f, self.x)
        self.B = sym.Jacobian(self.f, self.u)
        self.c = -(np.dot(self.A, self.x)+np.dot(self.B, self.u))+self.f

    def construct_linearized_system_at(self, env):
        self.A = sym.Jacobian(self.f, self.x)
        self.B = sym.Jacobian(self.f, self.u)
        self.c = -(np.dot(self.A, self.x)+np.dot(self.B, self.u))+self.f

    def evaluate_x_next(self, env):
        return sym.Evaluate(self.f, env)


class DTContinuousSystem:
    def __init__(self, f, x, u, initial_env=None, input_limits = None):
        '''
        Continuous dynamical system x_dot = f(x,u)
        :param f: A symbolic expression of the system dynamics.
        :param x: A list of symbolic variables. States.
        :param u: A list of symbolic variable. Inputs.
        :param initial_env: A dictionary "environment" specifying the initial state of the system
        :param input_limits: Input limits of the system
        '''
        self.dynamics = ContinuousDynamics(f,x,u)
        if initial_env is None:
            self.env = {}
            for x_i in self.dynamics.x:
                self.env[x_i] = 0
        else:
            self.env = initial_env
        if input_limits is None:
            self.input_limits = np.vstack([np.full(u.shape[0], -1e9),np.full(u.shape[0], 1e9)])
        else:
            self.input_limits = input_limits

    def foward_step(self, u=None, linearlize=False, modify_system=True, step_size = 1e-3, return_as_env = False):
        if not modify_system:
            new_env = self.env.copy()
        else:
            new_env = self.env
        if u is not None:
            for i in range(u.shape[0]):
                new_env[self.dynamics.u[i]] = min(max(u[i],self.input_limits[0,i]),self.input_limits[1,i])
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

    def get_reachable_zonotope(self, state, step_size = 1e-2):
        current_linsys = self.get_linearization(state)
        u_bar = (self.input_limits[1,:]+self.input_limits[0,:])/2
        u_diff =(self.input_limits[1,:]-self.input_limits[0,:])/2
        # print(current_linsys.A, current_linsys.B, current_linsys.c)
        x = np.ndarray.flatten(np.dot(current_linsys.A*step_size+np.eye(current_linsys.A.shape[0]),state))+\
            np.dot(current_linsys.B*step_size, u_bar)+np.ndarray.flatten(current_linsys.c*step_size)
        x = np.atleast_2d(x).reshape(-1,1)
        assert(len(x)==len(state))
        G = np.atleast_2d(np.dot(current_linsys.B*step_size, np.diag(u_diff)))
        # print('x', x)
        # print('G', G)
        return zonotope(x,G)


    def get_linearization(self, state=None):
        if state is None:
            return self.dynamics.construct_linearized_system_at(self.env)
        else:
            env = self._state_to_env(state)
            return self.dynamics.construct_linearized_system_at(env)

    def _state_to_env(self, state):
        env = {}
        # print('state',state)
        for i, s_i in enumerate(state):
            env[self.dynamics.x[i]] = s_i
        for u_i in self.dynamics.u:
            env[u_i] = 0
        return env

    def get_current_state(self):
        return extract_variable_value_from_env(self.dynamics.x, self.env)

class DTHybridSystem:
    def __init__(self, f_list, x, u, c_list, initial_env=None, input_limits=None):
        '''
        Hybrid system with multiple dynamics modes
        :param f_list: numpy array of system dynamics modes
        :param x:
        :param u:
        :param c_list: numpy array of Pydrake symbolic expressions c_i(x,u) describing when the system belong in that mode.
                        c_i(x,u) is a vector of functions describing the mode. All of c_i(x,u) >= 0 when system is in mode i.
                        Modes should be complete and mutually exclusive.
        :param initial_env: A dictionary "environment" specifying the initial state of the system
        :param input_limits: Input limits of the system
        '''
        assert f_list.shape[0] == c_list.shape[0]
        self.mode_count = f_list.shape[0]
        self.dynamics_list = np.asarray([ContinuousDynamics(f,x,u) for f in f_list])
        if initial_env is None:
            self.env = {}
            for x_i in self.dynamics_list.x:
                self.env[x_i] = 0
        else:
            self.env = initial_env
        if input_limits is None:
            self.input_limits = np.vstack([np.full(u.shape[0], -1e9),np.full(u.shape[0], 1e9)])
        else:
            self.input_limits = input_limits
        self.c_list = c_list
        # Check the mode the system is in
        self.current_mode = -1
        #TODO

    def foward_step(self, u=None, linearlize=False, modify_system=True, step_size = 1e-3, return_as_env = False,
                    return_mode = False):
        if not modify_system:
            new_env = self.env.copy()
        else:
            new_env = self.env
        if u is not None:
            for i in range(u.shape[0]):
                new_env[self.dynamics_list.u[i]] = min(max(u[i],self.input_limits[0,i]),self.input_limits[1,i])
        else:
            for i in range(self.dynamics_list.u.shape[0]):
                new_env[self.dynamics_list.u[i]] = 0

        # Check for which mode the system is in
        delta_x = None
        x_new = None
        mode = -1
        for i, c_i in enumerate(self.c_list):
            if sym.Evaluate(c_i(new_env)).all() >= 0:
                if self.dynamics_list[i].type == 'continuous':
                    delta_x = self.dynamics_list[i].evaluate_xdot(new_env, linearlize)*step_size
                elif self.dynamics_list[i].type == 'discrete':
                    x_new = self.dynamics_list[i].evaluate_x_next(new_env, linearlize)
                else:
                    raise ValueError
                mode = i
                break
        assert(mode != -1) # The system should always be in one mode
        #FIXME: check if system is in 2 modes (illegal)

        #assign new xs
        if self.dynamics_list[mode].type=='continuous':
            for i in range(delta_x.shape[0]):
                new_env[self.dynamics_list[mode].x[i]] += delta_x[i]
        elif self.dynamics_list[mode].type=='discrete':
            for i in range(x_new.shape[0]):
                new_env[self.dynamics_list[mode].x[i]] += x_new[i]
        else:
            raise ValueError

        #return options
        if return_as_env and not return_mode:
            return new_env
        elif return_as_env and return_mode:
            return new_env, mode
        elif not return_as_env and not return_mode:
            return extract_variable_value_from_env(self.dynamics_list[mode].x, new_env)
        else:
            return extract_variable_value_from_env(self.dynamics_list[mode].x, new_env), mode

    def get_reachable_zonotopes(self, state, step_size=1e-2):
        zonotopes_list = []
        for mode, c_i in enumerate(self.c_list):
            # FIXME: check if the mode is reachable
            current_linsys = self.get_linearization(mode, state)
            u_bar = (self.input_limits[1, :] + self.input_limits[0, :]) / 2
            u_diff = (self.input_limits[1, :] - self.input_limits[0, :]) / 2
            # print(current_linsys.A, current_linsys.B, current_linsys.c)
            if self.dynamics_list[mode].type == 'continuous':
                x = np.ndarray.flatten(
                    np.dot(current_linsys.A * step_size + np.eye(current_linsys.A.shape[0]), state)) + \
                    np.dot(current_linsys.B * step_size, u_bar) + np.ndarray.flatten(current_linsys.c * step_size)
                x = np.atleast_2d(x).reshape(-1, 1)
                assert (len(x) == len(state))
                G = np.atleast_2d(np.dot(current_linsys.B * step_size, np.diag(u_diff)))
                # print('x', x)
                # print('G', G)

            elif self.dynamics_list[mode].type == 'discrete':
                x = np.ndarray.flatten(
                    np.dot(current_linsys.A, state)) + \
                    np.dot(current_linsys.B, u_bar) + np.ndarray.flatten(current_linsys.c)
                x = np.atleast_2d(x).reshape(-1, 1)
                assert (len(x) == len(state))
                G = np.atleast_2d(np.dot(current_linsys.B, np.diag(u_diff)))
                # print('x', x)
                # print('G', G)
            else:
                raise ValueError

            zonotopes_list.append(zonotope(x,G))

        return np.asarray(zonotopes_list)

    def get_linearization(self, mode, state=None):
        if state is None:
            return self.dynamics_list[mode].construct_linearized_system_at(self.env)
        else:
            env = self._state_to_env(mode, state)
            return self.dynamics_list[mode].construct_linearized_system_at(env)

    def _state_to_env(self, mode, state):
        env = {}
        # print('state',state)
        for i, s_i in enumerate(state):
            env[self.dynamics_list[mode].x[i]] = s_i
        for u_i in self.dynamics_list[mode].u:
            env[u_i] = 0
        return env

    def get_current_state(self):
        return extract_variable_value_from_env(self.dynamics_list[self.current_mode].x, self.env)
