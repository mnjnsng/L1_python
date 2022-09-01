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

# def f(state):
#     gravity = 9.8
#     masscart = 1.0
#     masspole = 0.1
#     total_mass = masspole + masscart
#     length = 0.5  
#     polemass_length = masspole * length
#     force_mag = 10.0
#     next_state=np.zeros(state.shape)
#     x, x_dot, theta, theta_dot = state 

#     costheta = math.cos(theta)
#     sintheta = math.sin(theta)

#     temp = (polemass_length*theta_dot*theta_dot * sintheta) / total_mass

#     theta_ddot= (gravity * sintheta - costheta * temp) / (
#                 length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
#     x_ddot= temp - polemass_length*theta_ddot*costheta / total_mass

#     next_state=(x_dot,x_ddot,theta_dot,theta_ddot)
#     next_state = np.array(next_state)
#     return next_state

# # def g(state):
    
#     gravity = 9.8
#     masscart = 1.0
#     masspole = 0.1
#     total_mass = masspole + masscart
#     length = 0.5  
#     polemass_length = masspole * length
#     force_mag = 10.0
#     next_state=np.zeros(state.shape)
#     x, x_dot, theta, theta_dot = state
#     costheta = math.cos(theta)
#     sintheta = math.sin(theta)
#     x_ddot= force_mag/total_mass
#     theta_ddot= -costheta*force_mag /(
#                 length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))

#     next_state=(0,x_ddot,0, theta_ddot)
#     next_state = np.array(next_state)
#     return np.expand_dims(next_state, axis = -1)
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

    # temp = (self.polemass_length*theta_dot*theta_dot * sintheta) / self.total_mass

    # theta_ddot= (self.gravity * sintheta - costheta * temp) / (
    #             self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
    # x_ddot= temp - self.polemass_length*theta_ddot*costheta / self.total_mass
    temp = (polemass_length*theta_dot*theta_dot * sintheta) / total_mass

    theta_acc= (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))

    theta_ddot= (gravity * sintheta - costheta * polemass_length*theta_dot*theta_dot*sintheta/total_mass) / (
                length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
    x_ddot= polemass_length*theta_dot*theta_dot*sintheta/total_mass - polemass_length*theta_acc*costheta / total_mass

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
    # x_ddot= self.force_mag/self.total_mass
    # theta_ddot= -costheta*self.force_mag /(
    #             self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
    theta_ddot = -(costheta/total_mass)/(length*(4/3-masspole*costheta*costheta/total_mass))
    x_ddot = 1/total_mass+polemass_length*theta_ddot*costheta/total_mass
    next_state=(0,x_ddot* force_mag,0, theta_ddot* force_mag) 
    next_state = np.array(next_state)
    return np.expand_dims(next_state, axis = -1)

adaptive_controller = L1_adapt(env, f, g, observation)

obs_list=[]
policy=[]
matched_uncertainty = []
um1 = []
um2 = []
um3 = []
prediction = []

for _ in range(700):
    # frames.append(env.render(mode="rgb_array"))
    
    u_bl= lqr_policy(observation)
    u, sigma_hat_m,sigma_hat_um,thetahat =adaptive_controller.get_control_input(observation,u_bl)
    # u = u_bl    #uncomment to run baseline controller(lqr)

    observation=adaptive_controller.plant(observation,np.expand_dims(u,axis=-1))

    obs_list.append(observation[2])
    policy.append(u.squeeze(0))
    matched_uncertainty.append(sigma_hat_m)
    um1.append(sigma_hat_um[0])
    um2.append(sigma_hat_um[1])
    um3.append(sigma_hat_um[2])
    prediction.append(thetahat)

# env.close()

t = np.linspace(0,700*0.02,700)
plt.plot(t,obs_list,'r',t,prediction,'--')
plt.xlabel('Time (t)')
plt.ylabel(r'Angle (in radians)')
plt.legend([r'$\theta$',r'$\theta_hat$'])
plt.savefig('Trajectory_lqr.png', format = 'png')
plt.show()

# plt.plot(t,policy,'r')
# plt.xlabel('Time (t)')
# plt.ylabel('Control input (u)')
# plt.legend([r'$u$'])
# plt.savefig('Control-input_lqr.png', format = 'png')

sigma_m = 0.5*np.sin(0.8*t)
plt.plot(t,matched_uncertainty,'r', t,sigma_m,'b')
plt.xlabel('Time (t)')
plt.ylabel('sigma_hat_m ')
plt.legend([r'$\hat{\sigma}_m$',r'$\sigma_m$'])
plt.savefig('Matched_uncertainty.png', format = 'png')
plt.show()

sigma_um1 = 0.1*np.sin(0.5*t)
# sigma_um2 = [0]*len(t)
# sigma_um3 = [0] * len(t)
# plt.plot(t,um1,'r', t,um2,'b', t,um3,'g', t,sigma_um1,'--r', t,sigma_um2,'--b', t,sigma_um3, '--g')
plt.plot(t,um1,'r', t,sigma_um1,'--r')
plt.xlabel('Time (t)')
plt.ylabel('sigma_hat_um ')
plt.legend([r'$\hat{\sigma}_{um}$',r'$\sigma_{um}$'])
plt.savefig('Unmatched_uncertainty.png', format = 'png')

plt.show()

# save_frames_as_gif(frames)



