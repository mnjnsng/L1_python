import gym
import math 
import gym_cartpole_continuous
import control
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from l1_adaptive_controller import L1_adapt


def save_frames_as_gif(frames, path='./', filename='cartpole_lqr_l1.gif'):

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0,
               frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def lqr_policy(observation):

    # cost function

    Q = np.identity(4)
    R = np.identity(1)

    # linearization
    A = np.array([[0, 1, 0, 0], [0, 0, -0.98, 0],
                 [0, 0, 0, 1], [0, 0, 21.56, 0]])

    B = np.array([[0, 1, 0, -2]]).T

    K, S, E = control.lqr(A, B, Q, R)

    action = -1*np.dot(K, observation)

    if action >= 1:
        return np.array([1])
    elif action <= -1:
        return np.array([-1])
    else:
        return action


env = gym.make('CartPoleContinuous-v0')

M = float(env.masscart)
m = float(env.masspole)
l = float(env.length)
g = float(env.gravity)

frames = []
observation = env.reset()

# def f(x):
#     Am = np.array([[0, 1, 0, 0], [0, 0, -0.98, 0],
#                   [0, 0, 0, 1], [0, 0, 21.56, 0]])
#     return np.matmul(Am, x)


# def g(x):
#     Bm = np.array([[0, 1, 0, -2]]).T
#     return Bm

def f(state):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5  
    polemass_length = masspole * length
    force_mag = 10.0
    next_state=np.zeros(state.shape)
    x, x_dot, theta, theta_dot = state 

    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (polemass_length*theta_dot*theta_dot * sintheta) / total_mass

    theta_ddot= (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
    x_ddot= temp - polemass_length*theta_ddot*costheta / total_mass

    next_state=(x_dot,x_ddot,theta_dot,theta_ddot)
    next_state = np.array(next_state)
    return next_state

def g(state):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5  
    polemass_length = masspole * length
    force_mag = 10.0
    next_state=np.zeros(state.shape)
    x, x_dot, theta, theta_dot = state
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    x_ddot= force_mag/total_mass
    theta_ddot= -costheta*force_mag /(
                length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))

    next_state=(0,x_ddot,0, theta_ddot)
    next_state = np.array(next_state)
    return np.expand_dims(next_state, axis = -1)


adaptive_controller = L1_adapt(env, f, g)

obs_list=[]
policy=[]

for _ in range(700):
    frames.append(env.render(mode="rgb_array"))
    
    u_bl= lqr_policy(observation)
    u =adaptive_controller.get_control_input(observation,u_bl)
    # u = u_bl    #uncomment to run baseline controller(lqr)

    observation=adaptive_controller.plant(observation,np.expand_dims(u,axis=-1))

    obs_list.append(observation[2])
    policy.append(u.squeeze(0))

env.close()

t = np.linspace(0,700*0.02,700)
plt.plot(t,obs_list)
plt.xlabel('Time (t)')
plt.ylabel(r'Angle (in radians)')
plt.legend([r'$\theta$'])
plt.savefig('Trajectory_lqr_l1.png', format = 'png')
plt.show()

plt.plot(t,policy,'r')
plt.xlabel('Time (t)')
plt.ylabel('Control input (u)')
plt.legend([r'$u$'])
plt.savefig('Control-input_lqr_l1.png', format = 'png')
plt.show()

save_frames_as_gif(frames)



