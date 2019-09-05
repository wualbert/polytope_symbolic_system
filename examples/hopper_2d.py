import pydrake.symbolic as sym
import numpy as np
from common.symbolic_system import DTHybridSystem, in_mode, extract_variable_value_from_env

class Hopper_2d(DTHybridSystem):
    def __init__(self, m=10, J=10, m_l=1, J_l=1, l1=0.0, l2=0.0, k_g=1e3, b_g=90,\
                 g=9.8, ground_height_function=lambda x: 0, initial_state=np.asarray([0.,0.,0.,1.5,1.0,0.,0.,0.,0.,0.])):


        '''
        2D hopper with actuated piston at the end of the leg.
        The model of the hopper follows the one described in "Hopping in Legged Systems" (Raibert, 1984)
        '''
        self.m = m
        self.J = J
        self.m_l = m_l
        self.J_l = J_l
        self.l1 = l1
        self.l2 = l2
        self.k_g = k_g
        self.b_g = b_g
        self.g = g
        self.ground_height_function = ground_height_function
        self.r0 = 5
        self.b_leg = 0
        # state machine for touchdown detection
        self.xTD = sym.Variable('xTD')
        self.was_in_contact = False

        # Symbolic variables
        # State variables are s = [x_ft, y_ft, theta, phi, r]
        # self.x = [s, sdot]
        # Inputs are self.u = [tau, chi]
        self.x = np.array([sym.Variable('x_' + str(i)) for i in range(10)])
        self.u = np.array([sym.Variable('u_' + str(i)) for i in range(2)])

        # Initial state
        self.initial_env = {}
        for i, state in enumerate(initial_state):
            self.initial_env[self.x[i]]=state
        self.initial_env[self.xTD] = 0
        self.k0 = 300
        # print(self.initial_env)

        # Dynamic modes
        Fx_contact = -self.k_g*(self.x[0]-self.xTD)-self.b_g*self.x[5]
        Fx_flight = 0.
        Fy_contact = -self.k_g*(self.x[1]-self.ground_height_function(self.x[0]))-self.b_g*self.x[6]
        Fy_flight = 0.

        R = self.x[4]-self.l1
        # EOM is obtained from Russ Tedrake's Thesis
        a1 = -self.m_l*R
        a2 = (self.J_l-self.m_l*R*self.l1)*sym.cos(self.x[2])
        b1 = self.m_l*R
        b2 = (self.J_l -self.m_l*R*self.l1)*sym.sin(self.x[2])
        c1 = self.m*R
        c2 = (self.J_l+self.m*R*self.x[4])*sym.cos(self.x[2])
        c3 = self.m*R*self.l2*sym.cos(self.x[3])
        c4 = self.m*R*sym.sin(self.x[2])
        d1 = -self.m*R
        d2 = (self.J_l+self.m*R*self.x[4])*sym.sin(self.x[2])
        d3 = self.m*R*self.l2*sym.sin(self.x[3])
        d4 = -self.m*R*sym.cos(self.x[2])
        e1 = self.J_l*self.l2*sym.cos(self.x[2]-self.x[3])
        e2 = -self.J*R
        F_leg_flight = -20*(self.x[4]-self.r0)-self.b_leg*3*self.x[9]
        F_leg_ascend = -self.k0*(self.x[4]-self.r0)-self.b_leg*self.x[9]
        F_leg_descend = -self.u[1]*(self.x[4]-self.r0)-self.b_leg*self.x[9]
        def get_ddots(Fx, Fy, F_leg, u0):
            alpha = (self.l1*Fy*sym.sin(self.x[2])-self.l1*Fx*sym.cos(self.x[2])-u0)
            A = sym.cos(self.x[2])*alpha-R*(Fx-F_leg*sym.sin(self.x[2])-self.m_l*self.l1*self.x[7]**2*sym.sin(self.x[2]))
            B = sym.sin(self.x[2])*alpha+R*(self.m_l*self.l1*self.x[7]**2*sym.cos(self.x[2])+Fy-F_leg*sym.cos(self.x[2])-self.m_l*self.g)
            C = sym.cos(self.x[2])*alpha+R*F_leg*sym.sin(self.x[2])+self.m*R*(self.x[4]*self.x[7]**2*sym.sin(self.x[2])+self.l2*self.x[8]**2*sym.sin(self.x[3])-2*self.x[9]*self.x[7]*sym.cos(self.x[2]))
            D = sym.sin(self.x[2])*alpha-R*(F_leg*sym.cos(self.x[2])-self.m*self.g)-self.m*R*(2*self.x[9]*self.x[7]*sym.sin(self.x[2])+self.x[4]*self.x[7]**2*sym.cos(self.x[2])+self.l2*self.x[8]**2*sym.cos(self.x[3]))
            E = self.l2*sym.cos(self.x[2]-self.x[3])*alpha-R*(self.l2*F_leg*sym.sin(self.x[3]-self.x[2])+u0)

            return np.asarray([(A*b1*c2*d4*e2 - A*b1*c3*d4*e1 - A*b1*c4*d2*e2 + A*b1*c4*d3*e1 + A*b2*c4*d1*e2 - B*a2*c4*d1*e2 - C*a2*b1*d4*e2 + D*a2*b1*c4*e2 + E*a2*b1*c3*d4 - E*a2*b1*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2),\
            (A*b2*c1*d4*e2 + B*a1*c2*d4*e2 - B*a1*c3*d4*e1 - B*a1*c4*d2*e2 + B*a1*c4*d3*e1 - B*a2*c1*d4*e2 - C*a1*b2*d4*e2 + D*a1*b2*c4*e2 + E*a1*b2*c3*d4 - E*a1*b2*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2),\
            -(A*b1*c1*d4*e2 - B*a1*c4*d1*e2 - C*a1*b1*d4*e2 + D*a1*b1*c4*e2 + E*a1*b1*c3*d4 - E*a1*b1*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2),\
            (A*b1*c1*d4*e1 - B*a1*c4*d1*e1 - C*a1*b1*d4*e1 + D*a1*b1*c4*e1 + E*a1*b1*c2*d4 - E*a1*b1*c4*d2 + E*a1*b2*c4*d1 - E*a2*b1*c1*d4)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2),\
            (A*b1*c1*d2*e2 - A*b1*c1*d3*e1 - A*b2*c1*d1*e2 - B*a1*c2*d1*e2 + B*a1*c3*d1*e1 + B*a2*c1*d1*e2 - C*a1*b1*d2*e2 + C*a1*b1*d3*e1 + C*a1*b2*d1*e2 + D*a1*b1*c2*e2 - D*a1*b1*c3*e1 - D*a2*b1*c1*e2 - E*a1*b1*c2*d3 + E*a1*b1*c3*d2 - E*a1*b2*c3*d1 + E*a2*b1*c1*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2)])

        flight_dynamics = np.hstack((self.x[5:], get_ddots(Fx_flight, Fy_flight, F_leg_flight, self.u[0])))
        contact_descend_dynamics = np.hstack((self.x[5:], get_ddots(Fx_contact, Fy_contact, F_leg_descend, self.u[0])))
        contact_ascend_dynamics = np.hstack((self.x[5:], get_ddots(Fx_contact, Fy_contact, F_leg_ascend, self.u[0])))

        flight_conditions = np.asarray([self.x[1] > self.ground_height_function(self.x[0])])
        contact_descend_coditions = np.asarray([self.x[1] <= self.ground_height_function(self.x[0]), self.x[9] < 0])
        contact_ascend_coditions = np.asarray([self.x[1] <= self.ground_height_function(self.x[0]), self.x[9] >= 0])

        self.f_list = np.asarray([flight_dynamics, contact_descend_dynamics, contact_ascend_dynamics])
        self.f_type_list = np.asarray(['continuous', 'continuous', 'continuous'])
        self.c_list = np.asarray([flight_conditions, contact_descend_coditions, contact_ascend_coditions])

        DTHybridSystem.__init__(self, self.f_list, self.f_type_list, self.x, self.u, self.c_list, \
                                self.initial_env, input_limits=np.vstack([[-10,10], [10,self.k0-1]]))

    def get_cg_coordinate_states(self, env = None):
        """
        Convert the state into the representation used in MIT 6.832 PSet4
        [x2, y2, theta2, theta1-theta2, w]
        :param env:
        :return:
        """
        if env is None:
            env = self.env
        # extract variables from the environment
        theta1 = env[self.x[0]]
        theta2 = env[self.x[1]]
        x0 = env[self.x[2]]
        y0 = env[self.x[3]]
        w = env[self.x[4]]
        theta1_dot = env[self.x[5]]
        theta2_dot = env[self.x[6]]
        x0_dot = env[self.x[7]]
        y0_dot = env[self.x[8]]
        w_dot =  env[self.x[9]]

        # compute forward kinematics
        x1 = x0+self.r1*np.sin(theta1)
        y1 = y0+self.r1*np.cos(theta1)
        x2 = x0+w*np.sin(theta1)+self.r2*np.cos(theta2)
        y2 = y0+w*np.cos(theta1)+self.r2*np.sin(theta2)

        # compute derivatives
        x1_dot = x0_dot+self.r1*np.cos(theta1)*theta1_dot
        y1_dot = y0_dot-self.r1*np.sin(theta1)*theta1_dot
        x2_dot = x0_dot+w_dot*np.sin(theta1)+w*np.cos(theta1)*theta1_dot+self.r2*np.cos(theta2)*theta2_dot
        y2_dot = y1_dot+w_dot*np.cos(theta1)-w*np.sin(theta1)*theta1_dot-self.r2*np.sin(theta2)*theta2_dot

        return x2, y2, theta2, theta1-theta2, w, x2_dot, y2_dot, theta2_dot, theta1_dot-theta2_dot, w_dot

    def do_internal_updates(self):
        # extract variables from the environment
        try:
            x0 = self.env[self.x[0]][0]
        except:
            x0 = self.env[self.x[0]]
        try:
            y0 = self.env[self.x[1]][0]
        except:
            y0 = self.env[self.x[1]]
        if not self.was_in_contact and y0-self.ground_height_function(x0)<=0:
            # just touched down
            # set the touchdown point
            self.env[self.xTD] = x0
        self.was_in_contact=(y0-self.ground_height_function(x0)<=0)
        #FIXME: _state_to_env does not set self.env[self.xTD]

    def _state_to_env(self, state, u=None):
        env = {}
        # print('state',state)
        for i, s_i in enumerate(state):
            env[self.x[i]] = s_i
        if u is None:
            for u_i in self.u:
                env[u_i] = 0
        else:
            for i, u_i in enumerate(u):
                env[self.u[i]] = u[i]
        # FIXME: This is sketchy
        env[self.xTD] = state[0]
        return env

    def forward_step(self, u=None, linearlize=False, modify_system=True, step_size = 1e-3, return_as_env = False,
                     return_mode = False, starting_state=None):
        if starting_state is not None:
            new_env = self._state_to_env(starting_state, u)
        elif not modify_system:
            new_env = self.env.copy()
        else:
            new_env = self.env
        if u is not None:
            for i in range(u.shape[0]):
                new_env[self.u[i]] = min(max(u[i],self.input_limits[0,i]),self.input_limits[1,i])
        else:
            for i in range(self.u.shape[0]):
                new_env[self.u[i]] = 0
        # Check for which mode the system is in
        delta_x = None
        x_new = None
        mode = -1
        for i, c_i in enumerate(self.c_list):
            is_in_mode = in_mode(c_i,new_env)
            if not is_in_mode:
                continue
            if self.dynamics_list[i].type == 'continuous':
                delta_x = self.dynamics_list[i].evaluate_xdot(new_env, linearlize)*step_size
            elif self.dynamics_list[i].type == 'discrete':
                x_new = self.dynamics_list[i].evaluate_x_next(new_env, linearlize)
            else:
                raise ValueError
            mode = i
            break
        assert(mode != -1) # The system should always be in one mode
        # print('mode', mode)
        # print(self.env)
        #FIXME: check if system is in 2 modes (illegal)

        #assign new xs
        if self.dynamics_list[mode].type=='continuous':
            for i in range(delta_x.shape[0]):
                new_env[self.x[i]] += delta_x[i]
        elif self.dynamics_list[mode].type=='discrete':
            for i in range(x_new.shape[0]):
                new_env[self.x[i]] = x_new[i]
        else:
            raise ValueError

        # if new_env[self.x[4]] < self.r_min or new_env[self.x[4]]>self.r_max:
        #     new_env[self.x[4]] = min(max(new_env[self.x[4]], self.r_min), self.r_max)
        #     new_env[self.x[9]] = 0
        self.do_internal_updates()

        #return options
        if return_as_env and not return_mode:
            return new_env
        elif return_as_env and return_mode:
            return new_env, mode
        elif not return_as_env and not return_mode:
            return extract_variable_value_from_env(self.x, new_env)
        else:
            return extract_variable_value_from_env(self.x, new_env), mode

