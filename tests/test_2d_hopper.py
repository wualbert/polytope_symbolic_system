from examples.hopper_2d import Hopper_2d
import matplotlib.pyplot as plt
import numpy as np

def test_raibert_controller_hopper():
    initial_state = np.asarray([0.,0.,0.0,1.5,1.5,0.,0.,0.6,0.,0.])
    hopper = Hopper_2d(initial_state=initial_state)
    # simulate the hopper
    state_count = 100000
    states = np.zeros([10, state_count])
    states[:, 0] = initial_state
    cg_coordinate_states = np.zeros([10, state_count])
    end_count = None
    try:
        for i in range(1,state_count):
            if i%10000==0:
                print('iteration %i' %i)
            # do forward kinematics for body position
            chi = 1.0
            # in contact
            if states[3, i-1]<=0:
                if hopper.get_cg_coordinate_states()[6]>0:
                    chi = 1.05
            tau = 0.
            states[:, i] = hopper.forward_step(step_size=1e-4, u=np.asarray([tau, chi]))
            cg_coordinate_states[:, i] = hopper.get_cg_coordinate_states()
    except:
        end_count = i
        print('Simulation terminated at step %i' %i)
        print('Env is:', hopper.get_current_state())
    # plot the Raibert coordinate states
    fig1, ax1 = plt.subplots(5,1)
    labels1 = ['$\\theta_1$', '$\\theta_2$', '$x_0$', '$y_0$', 'w']
    for i in range(5):
        ax1[i].plot(states[i,1:end_count])
        ax1[i].set_xlabel('Steps')
        ax1[i].set_ylabel(labels1[i])
    # plot the cg coordinate states
    fig2, ax2 = plt.subplots(5,1)
    labels2 = ['$x$', '$y$', '$\\theta$', '$\\alpha$', 'l']
    for i in range(5):
        ax2[i].plot(cg_coordinate_states[i,1:end_count])
        ax2[i].set_xlabel('Steps')
        ax2[i].set_ylabel(labels2[i])
    plt.show()


if __name__=='__main__':
    test_raibert_controller_hopper()