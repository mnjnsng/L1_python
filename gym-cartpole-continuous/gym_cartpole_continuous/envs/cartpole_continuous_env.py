import math
from scipy.linalg import null_space, inv, expm
import numpy as np
from gym import spaces, logger
from gym.envs.classic_control import CartPoleEnv


class CartPoleContinuousEnv(CartPoleEnv):
    def __init__(self):
        super().__init__()

        # direction & scale of force magnitude.
        self.min_action = np.float32(-1.0)
        self.max_action = np.float32(1.0)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), )
        self.time = 0.0

    def f(self,state):
        next_state=np.zeros(state.shape)
        x, x_dot, theta, theta_dot = state 

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # temp = (self.polemass_length*theta_dot*theta_dot * sintheta) / self.total_mass

        # theta_ddot= (self.gravity * sintheta - costheta * temp) / (
        #             self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        # x_ddot= temp - self.polemass_length*theta_ddot*costheta / self.total_mass
        temp = (self.polemass_length*theta_dot*theta_dot * sintheta) / self.total_mass

        theta_acc= (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))

        theta_ddot= (self.gravity * sintheta - costheta * self.polemass_length*theta_dot*theta_dot*sintheta/self.total_mass) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        x_ddot= self.polemass_length*theta_dot*theta_dot*sintheta/self.total_mass - self.polemass_length*theta_acc*costheta / self.total_mass

        next_state=(x_dot,x_ddot,theta_dot,theta_ddot)
        next_state = np.array(next_state)[:,np.newaxis]
        return next_state


    def g(self,state):
        next_state=np.zeros(state.shape)
        x, x_dot, theta, theta_dot = state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        # x_ddot= self.force_mag/self.total_mass
        # theta_ddot= -costheta*self.force_mag /(
        #             self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        theta_ddot = -(costheta/self.total_mass)/(self.length*(4/3-self.masspole*costheta*costheta/self.total_mass))
        x_ddot = 1/self.total_mass+self.polemass_length*theta_ddot*costheta/self.total_mass
        next_state=(0,x_ddot* self.force_mag,0, theta_ddot* self.force_mag) 
        next_state = np.array(next_state)
        return np.expand_dims(next_state, axis = -1)



    def step(self,action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        act = action[0] #+ 0.5*np.sin(0.8*self.time) 
        
        if act > self.max_action:
            act = self.max_action
        elif act < self.min_action:
            act = self.min_action

        # force = self.force_mag * act
        force = act+0.5*np.sin(0.8*self.time)

        
        state = self.state
        x, x_dot, theta, theta_dot = state
        sigma_um = np.array([0.1*np.sin(0.5*self.time),0,0])
        self.time += self.tau
        state_dot = self.f(state) + self.g(state) * force + np.expand_dims(null_space(self.g(state).T)@sigma_um,-1)
        x_dot, xacc , theta_dot, thetaacc = state_dot.squeeze() 
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        
        #x, x_dot, theta, theta_dot = new_state.squeeze()

        self.state = (x, x_dot, theta, theta_dot)


        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True."
                            " You should always call 'reset()' once you receive 'done = True' -- any further steps "
                            "are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0


        return np.array(self.state), reward, done, {}

    def step2(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Discrete Case:(just for reference)
        # force = self.force_mag if action == 1 else -self.force_mag

        # Continuous Case:
        act = action[0] #+ np.sin(0.8*self.time) 
        self.time += self.tau
        if act > self.max_action:
            act = self.max_action
        elif act < self.min_action:
            act = self.min_action

        force = self.force_mag * act
        # Note: everything below this is same as gym's cartpole step fun.
        
        state = self.state
        x, x_dot, theta, theta_dot = state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True."
                            " You should always call 'reset()' once you receive 'done = True' -- any further steps "
                            "are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}


if __name__ == "__main__":
    # from l1_adaptive_controller import L1_adapt
    import gym
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

    env = CartPoleContinuousEnv()
    observation1 = env.reset()
    action = np.array([-0.3])
    s1,_,_,_ = env.step(action)
    
    env2 = CartPoleContinuousEnv()
    observation2 = env2.reset()
    s2,_,_,_ = env.step2(action)
    print(s1)
    print(s2)
    
