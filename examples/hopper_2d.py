import pydrake.symbolic as sym
from common.symbolic_system import *

class Hopper_2d(DTHybridSystem):
    def __init__(self, M1, I1, M2, I2, r1, r2, w, KL, KL2, BL2, KG, BG, k0, g, ground_height_function, initial_state):


        '''
        2D hopper with actuated piston at the end of the leg.
        The model of the hopper follows the one described in "Hopping in Legged Systems" (Raibert, 1984)
        '''
        self.M1 = M1
        self.I1 = I1
        self.M2 = M2
        self.I2 = I2
        self.r1 = r1
        self.r2 = r2
        self.w = w
        self.KL = KL
        self.KL2 = KL2
        self.BL2 = BL2
        self.k0 = k0
        self.KG = KG
        self.BG = BG
        self.g = g
        self.xTD = 0
        self.ground_height_function = ground_height_function

        # Symbolic variables
        # State variables are s = [theta1, theta2, x0, y0, w]
        # self.x = [s, sdot]
        # Inputs are self.u = [tau, chi]
        self.x = np.array([sym.Variable('x_' + str(i)) for i in range(10)])
        self.u = np.array([sym.Variable('u_' + str(i)) for i in range(2)])

        # Initial state
        self.initial_env = {}
        for state, i in enumerate(initial_state):
            self.initial_env[self.x[i]]=state

        # Dynamic modes
        FK_extended = self.KL*(self.k0-self.x[4]+self.u[1])
        FK_retracted = self.KL2*(self.k0-self.x[4]+self.u[1])-self.BL2*self.x[9]
        FX_contact = -self.KG*(self.x[2]-self.xTD)-self.BG*self.x[7] #FIXME: handling xTD
        FX_flight = 0
        FY_contact = -self.KG(self.x[3]-self.ground_height_function(self.x[2]))-self.BG*(self.x[8])
        FY_flight = 0

        W = self.x[4]-self.r1

        # EOM is obtained from Raibert 1984 paper, Appendix I eqn 40~44
        # The EOM is in the following form
        # a1*theta1_ddot + a2*theta2_ddot + a3*x0_ddot + a4*w_ddot = A
        # b1*theta1_ddot + b2*theta2_ddot + b3*y0_ddot + b4*w_ddot = B
        # c1*theta1_ddot + c2*x0_ddot = C
        # d1*theta1_ddot + d2*y0_ddot = D
        # e1*theta1_ddot + e2*theta2_ddot = E

        a1 = sym.cos(self.x[0])*(self.M2*W*self.x[4]+self.I1)
        a2 = self.M2*self.r2*W*sym.cos(self.x[1])
        a3 = self.M2*W
        a4 = self.M2*W*sym.sin(self.x[0])

        b1 = -sym.sin(self.x[0])*(self.M2*W*self.x[4]+self.I1)
        b2 = -self.M2*self.r2*W*sym.sin(self.x[1])
        b3 = self.M2*W
        b4 = self.M2*W*sym.cos(self.x[1])

        c1 = sym.cos(self.x[0])*(self.M1*self.r1*W-self.I1)
        c2 = self.M1*W

        d1 = -sym.sin(self.x[0])*(self.M1*self.r1*W-self.I1)
        d2 = self.M1*W

        e1 = -sym.cos(self.x[1]-self.x[0])*self.I1*self.r2
        e2 = self.I2*self.x[4]

        # free flight with extended leg
        flight_extended_conditions = np.asarray([self.k0-self.x[4]+self.u[1]>0,
                                                      self.x[3]-self.ground_height_function(self.x[2])>0])

        # free flight with retracted leg
        flight_retracted_conditions = np.asarray([self.k0-self.x[4]+self.u[1]<=0,
                                                      self.x[3]-self.ground_height_function(self.x[2])>0])

        # contact with extended leg
        contact_extended_conditions = np.asarray([self.k0-self.x[4]+self.u[1]>0,
                                                      self.x[3]-self.ground_height_function(self.x[2])<=0])

        # contact with retracted leg
        contact_retracted_conditions = np.asarray([self.k0-self.x[4]+self.u[1]<=0,
                                                      self.x[3]-self.ground_height_function(self.x[2])<=0])


        # free flight
        free_flight_dynamics = np.asarray([self.x[1], -self.g])
        # piston contact
        piston_contact_dynamics = np.asarray([self.x[1], self.u[0]/self.m-self.g])
        # piston retract
        piston_retracted_dynamics = np.asarray([self.l+self.epsilon, -self.x[1]*self.b])
        self.f_list = np.asarray([free_flight_dynamics, piston_contact_dynamics, piston_retracted_dynamics])
        self.f_type_list = np.asarray(['continuous', 'continuous', 'discrete'])

        # contact mode conditions
        free_flight_conditions = np.asarray([self.x[0]>self.l+self.p])
        piston_contact_conditions = np.asarray([self.l<self.x[0], self.x[0]<=self.l+self.p])
        piston_retracted_conditions = np.asarray([self.x[0]<=self.l])
        self.c_list = np.asarray([free_flight_conditions, piston_contact_conditions, piston_retracted_conditions])

        DTHybridSystem.__init__(self, self.f_list, self.f_type_list, self.x, self.u, self.c_list, \
                                self.initial_env, np.asarray([[0], [self.f_max]]))
