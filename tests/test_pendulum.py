from examples.pendulum import *
import matplotlib.pyplot as plt

def test_static_pendulum():
    pend = Pendulum(initial_state = np.array([np.pi/2,0]), b=0.2)
    #simulate the pendulum
    state_count = 4000
    states = np.zeros([2, state_count])
    states[:,0] = np.array([np.pi/2,0])
    for i in range(state_count):
        states[:,i] = pend.foward_step(step_size=1e-2)
    plt.subplot(211)
    plt.plot(states[0,:])
    plt.xlabel('Steps')
    plt.ylabel('$\\theta$')
    plt.subplot(212)
    plt.plot(states[1,:])
    plt.xlabel('Steps')
    plt.ylabel('$\dot{\\theta}$')
    plt.show()

def test_controlled_pendulum():
    pend = Pendulum(initial_state = np.array([np.pi/2,0]), b=0.2,input_limits=np.array([[-.98],[.98]]))
    #simulate the pendulum for 200 steps
    state_count = 4000
    states = np.zeros([2, state_count])
    states[:,0] = np.array([np.pi/2.2,0])
    #Use PID controller to drive the pendulum upright
    for i in range(state_count):
        current_state = pend.get_current_state()
        u = 10*(current_state[0]-np.pi)+1*(current_state[1])
        states[:,i] = pend.foward_step(step_size=1e-2, u=np.asarray([-u]))
    plt.subplot(211)
    plt.plot(states[0,:])
    plt.xlabel('Steps')
    plt.ylabel('$\\theta$')
    plt.subplot(212)
    plt.plot(states[1,:])
    plt.xlabel('Steps')
    plt.ylabel('$\dot{\\theta}$')
    plt.show()

if __name__ == '__main__':
    test_static_pendulum()