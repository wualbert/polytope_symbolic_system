from examples.hopper_2d import Hopper_2d
import matplotlib.pyplot as plt
import numpy as np

def test_raibert_controller_hopper():
    initial_state = np.asarray([0.,0.,0.,1.5,1.5,0.,0.,0.,0.,0.])
    hopper = Hopper_2d(initial_state=initial_state)
    # simulate the hopper
    state_count = 25000
    states = np.zeros([10, state_count])
    states[:, 0] = initial_state
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
    plt.subplot(211)
    plt.plot(states[0, :])
    plt.xlabel('Steps')
    plt.ylabel('$x$')
    plt.subplot(212)
    plt.plot(states[1, :])
    plt.xlabel('Steps')
    plt.ylabel('$\dot{x}$')
    plt.show()


if __name__=='__main__':
    test_raibert_controller_hopper()