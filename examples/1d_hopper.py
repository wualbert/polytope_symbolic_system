import pydrake.symbolic as sym
from common.symbolic_system import *

class Hopper_1d(DTHybridSystem):
    def __init__(self, m, l, p, b, g, f_range, initial_state = None):
        '''
        Vertical 1D hopper with actuated piston at the end of the leg.
        The hopper has 3 dynamics mode decided by the body height h:
        1. h>l+p: free flight. xddot = -g
        2. l<h<=l+p: piston in contact with the ground and the hopper may push itself up.
            xddot = f/m-g.
        3. h<=l: piston is fully retracted. The hopper bounces off the ground
            in an elastic collision with xdot(n+1) = -b*xdot(n), where 0<b<1

        :param m: mass of the hopper
        :param l: leg length of the hopper
        :param p: piston length of the hopper
        :param b: damping factor fo the ground
        :param g: gravity constant
        :param f_range: maximum force the piston can exert
        '''
        self.m = m
        self.l = l
        self.p = p
        self.b = b
        self.g = g
        self.f_range = f_range
        if initial_state is None:
            self.initial_state = np.asarray([self.l+self.p, 0])
        else:
            self.initial_state = self.initial_state
        # Symbolic variables
        self.x = np.array([sym.Variable('x_' + str(i)) for i in range(2)])
        self.u = np.array([sym.Variable('u_' + str(i)) for i in range(1)])

        # Dynamic modes

        DTHybridSystem.__init__()
