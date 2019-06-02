import pydrake.symbolic as sym
import numpy as np
from symbolic_system import *

def test_with_toy_system():
    x = np.array([sym.Variable('x')])
    u = np.array([sym.Variable('u')])
    parameters = np.array([sym.Variable('param_'+str(i)) for i in range(2)])
    f = np.atleast_2d([sym.pow(x[0],3)+sym.pow(u[0],3)])
    print('f:', f)
    dyn = ContinuousDynamics(f,x,u)
    print('A:', dyn.A)
    print('B:', dyn.B)
    print('c:', dyn.c)
    env = {x[0]:3,u[0]:4,parameters[0]:1, parameters[1]:2}
    print('Nonlinear x_dot:', dyn.evaluate_xdot(env))
    print('linearized x_dot:', dyn.evaluate_xdot(env, linearize=True))

    sys = DTContinuousSystem(f, x, u, initial_env={x[0]:3})
    print('Nonlinear forward step:', sys.foward_step(u=np.array([2]),step_size=1, modify_system=False))
    print('Linear forward step:', sys.foward_step(u=np.array([2]),linearlize=True, step_size = 1,modify_system=False))

if __name__ == '__main__':
    test_with_toy_system()
