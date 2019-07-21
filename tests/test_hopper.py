from examples.hopper_1d import *
import matplotlib.pyplot as plt

def test_static_hopper():
    initial_state = np.asarray([2, 0])
    hopper = Hopper_1d(f_max=0, initial_state=initial_state)
    #simulate the hopper
    state_count = 20000
    states = np.zeros([2, state_count])
    states[:,0] = initial_state
    for i in range(state_count):
        states[:,i] = hopper.foward_step(step_size=1e-3)
    plt.subplot(211)
    plt.plot(states[0,:])
    plt.xlabel('Steps')
    plt.ylabel('$x$')
    plt.subplot(212)
    plt.plot(states[1,:])
    plt.xlabel('Steps')
    plt.ylabel('$\dot{x}$')
    plt.show()

if __name__=='__main__':
    test_static_hopper()