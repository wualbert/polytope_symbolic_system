from examples.hopper_2d import *
import matplotlib.pyplot as plt

def test_raibert_controller_hopper():
    initial_state = np.asarray([2, 0])
    hopper = Hopper_2d(initial_state=initial_state)
    # simulate the hopper
    state_count = 25000
    states = np.zeros([2, state_count])
    states[:, 0] = initial_state
    for i in range(state_count):
        chi = 1.0 if states[3]<0 else 1.5
        tau =
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