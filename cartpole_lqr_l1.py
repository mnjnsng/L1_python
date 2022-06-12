#from mbbl.env.gym_env import cartpole
import gym
import gym_cartpole_continuous
import control
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from l1_adaptive_controller import L1_adapt

def save_frames_as_gif(frames, path='./', filename='lqr_l1.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def lqr_policy(observation):

    # cost function 

    Q = np.identity(4)
    R = np.identity(1)

    # linearization
    #A = np.identity(4)
    A = np.array([[0,1,0,0],[0,0,-0.98, 0],[0, 0, 0, 1],[0, 0, 21.56, 0]])

    #B = np.ones((4,1))
    B = np.array([[0,1,0,-2]]).T

    K, S, E = control.lqr(A,B,Q,R)

    action = -1*np.dot(K,observation)


    if action >= 1:
        return np.array([1])
    elif action <=-1:
        return np.array([-1])
    else:
    	return action

env  = gym.make('CartPoleContinuous-v0')

M = float(env.masscart)
m = float(env.masspole)
l = float(env.length)
g = float(env.gravity)

frames = []
observation = env.reset()
total_reward = 0


def f(x):
     Am = np.array([[0,1,0,0],[0,0,-0.98, 0],[0, 0, 0, 1],[0, 0, 21.56, 0]])
     return np.matmul(Am,x)

def g(x):
    Bm = np.array([[0,1,0,-2]]).T
    return Bm

adaptive_controller = L1_adapt(env,f,g)

for _ in range(10):
    frames.append(env.render(mode="rgb_array"))
    observation, reward, done, info = env.step(env.action_space.sample())
    total_reward += reward

for _ in range(500):
    frames.append(env.render(mode="rgb_array"))
    
    u_bl= lqr_policy(observation)

    u=adaptive_controller.get_control_input(observation,u_bl)

    observation, reward, done, info = env.step(u_bl)
    total_reward += reward

env.close()
#save_frames_as_gif(frames) To save as gif