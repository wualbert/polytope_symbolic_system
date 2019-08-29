from examples.hopper_2d import Hopper_2d
import matplotlib.pyplot as plt
import numpy as np

def test_raibert_controller_hopper():
    initial_state = np.asarray([0.,0.,0.,1.5,1.5,0.,0.,0.,0.,0.])
    hopper = Hopper_2d(initial_state=initial_state)
    # simulate the hopper
    state_count = 1000
    states = np.zeros([10, state_count])
    states[:, 0] = initial_state
    center_states = np.zeros([4, state_count])
    for i in range(1,state_count):
        print('iteration %i' %i)
        # do forward kinematics for body position
        x1, y1, x2, y2, x1_dot, y1_dot, x2_dot, y2_dot = hopper.do_cg_forward_kinematics()
        chi = 1.0
        if states[3, i-1]<=0:
            # in contact
            if y2_dot>0:
                chi = 1.15
        tau = 0.
        states[:, i] = hopper.forward_step(step_size=1e-3, u=np.asarray([tau, chi]))
        center_states[:,i] = x2,y2,x2_dot,y2_dot
    fig1, ax1 = plt.subplots(2,1)
    ax1[0].plot(states[2, :])
    ax1[0].set_xlabel('Steps')
    ax1[0].set_ylabel('$x$')
    ax1[1].plot(states[3, :])
    ax1[1].set_xlabel('Steps')
    ax1[1].set_ylabel('$\dot{x}$')
    fig2, ax2 = plt.subplots()
    ax2.scatter(center_states[0,:], center_states[1,:], s=0.4)
    plt.show()


if __name__=='__main__':
    test_raibert_controller_hopper()