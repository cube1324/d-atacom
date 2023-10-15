
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box
from mushroom_rl.utils.viewer import Viewer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

class BallND(Environment):
    def __init__(self, n=2, horizon=500, gamma=0.99, n_sub_steps=4, center_const=False, acc_control=False, dense_const=False, return_cost=True):
        # Set the properties for spaces
        self.return_cost = return_cost
        self.n = n
        self.target_margin = 0.2  # Space between termination line and possible target
        self.agent_slack = 0.1    # Space between constraint violation and termination
        self.dt = 0.01
        self.respawn_interval = 200
        self.target_noise_std = 0.05
        self.enable_reward_shaping = False
        self.reward_shaping_slack = 0.1
        self.constr_dim = 1


        self.center_const = center_const
        self.acc_control = acc_control
        self.dense_const = dense_const

        self._n_sub_steps = n_sub_steps

        self._viewer = None
        self.background_img = None
        
        action_space = Box(low=-1, high=1, shape=(self.n,))
        if self.acc_control:
            observation_space = Box(low=0, high=1, shape=(3 * self.n,))
        else:
            observation_space = Box(low=0, high=1, shape=(2 * self.n,))

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        
        super().__init__(mdp_info)
        # Sets all the episode specific variables
        self.reset()
        
    def reset(self, state=None):
        if state is None:
            self._agent_position = 0.7 * np.ones(self.n,  dtype=np.float32)
        else:
            self._agent_position = state

        self._agent_velocity = np.zeros_like(self._agent_position)

        self._reset_target_location()
        self._current_step = 0
        return self.step(np.zeros(self.n))[0]
    
    def _get_reward(self):
        if self.enable_reward_shaping and self._is_agent_outside_shaping_boundary():
            return -1
        else:
            return np.maximum(1 - 10 * LA.norm(self._agent_position - self._target_position) ** 2, 0)
    
    def _reset_target_location(self):
        self._target_position = \
            (1 - 2 * self.target_margin) * np.random.random(self.n) + self.target_margin
    
    def _move_agent(self, action):
        for _ in range(self._n_sub_steps):
            if self.acc_control:
                self._agent_velocity += self.dt * action
                self._agent_position += self.dt * self._agent_velocity
            else:
                self._agent_position += self.dt * action
            # self._agent_position = np.clip(self._agent_position, 0, 1)
            # agent_velocity *= 0.95
    
    def _is_agent_outside_boundary(self):
        return np.any(self._agent_position < 0) or np.any(self._agent_position > 1)
    
    def _is_agent_outside_shaping_boundary(self):
        return np.any(self._agent_position < self.reward_shaping_slack) \
               or np.any(self._agent_position > 1 - self.reward_shaping_slack)

    def _update_time(self):
        # Assume that frequency of motor is 1 (one action per second)
        self._current_step += self._n_sub_steps
    
    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, self.target_noise_std, self.n)
    
    def get_num_constraints(self):
        return 2 * self.n

    def render(self, record=False):
        if self.n == 2:
            if self._viewer is None:
                self._viewer = Viewer(1, 1)

            if self.background_img is not None:
                self._viewer.background_image(self.background_img)

            self._viewer.circle(self._agent_position, radius=0.01, color=(255, 0, 0))
            self._viewer.circle(self._target_position, radius=0.01, color=(0, 0, 255))
            self._viewer.square(np.array([0.5, 0.5]), angle=0, edge=1 - 2 * self.agent_slack, width=1, color=(0, 255, 0))
            self._viewer.display(self.dt * self._n_sub_steps)

            self._viewer.circle(np.array([0.5, 0.5]), 0.1, color=(0, 255, 0), width=1)
        else:
            raise NotImplementedError()

    def get_constraint_values(self):
        # For any given n, there will be 2 * n constraints
        # a lower and upper bound for each dim
        # We define all the constraints such that C_i = 0
        # _agent_position > 0 + _agent_slack => -_agent_position + _agent_slack < 0
        min_constraints = self.agent_slack - self._agent_position
        # _agent_position < 1 - _agent_slack => _agent_position + agent_slack- 1 < 0
        max_constraint = self._agent_position + self.agent_slack - 1

        return np.concatenate([min_constraints, max_constraint])

    def step(self, action):
        # Check if the target needs to be relocated
        if self._current_step % self.respawn_interval == 0:
            self._reset_target_location()

        # Increment time
        self._update_time()

        clipped_action = np.clip(action, -1, 1)
        # Calculate new position of the agent
        self._move_agent(clipped_action)

        # Find reward         
        reward = self._get_reward()

        # Prepare return payload
        if self.acc_control:
            observation = np.concatenate([self._agent_position, self._agent_velocity, self._get_noisy_target_position()])
        else:
            observation = np.concatenate([self._agent_position, self._get_noisy_target_position()])

        done = self._is_agent_outside_boundary() \
               or self._current_step > self._mdp_info.horizon
            
        cost = 0
        if self.dense_const:
            cost_1 = self.agent_slack - self._agent_position
            cost_2 = self._agent_position - (1 - self.agent_slack)

            cost = max(*cost_1, *cost_2)

            if self.center_const:
                cost_3 = - np.sum((self._agent_position - 0.5) ** 2) + 0.1 ** 2
                cost = max(cost, cost_3)

        else:
            if np.any(self._agent_position < self.agent_slack) or np.any(self._agent_position > 1 - self.agent_slack):
                cost = 1

            if self.center_const:
                if np.sum((self._agent_position - 0.5) ** 2) < 0.1 ** 2:
                    cost = 1

        if self.return_cost:
            return observation, reward, cost, done, {'cost': cost}
        return observation, reward, done, {'cost': cost}


    @staticmethod
    def plot_constraint(_plot_constraint, save_path=None, debug=False):

        def get_plt_states(states):
            return (*states.T[:2],)

        N = 200
        X, Y = np.mgrid[0:1:complex(0, N), 0:1:complex(0, N)]
        states = np.vstack([X.ravel(), Y.ravel()]).T

        states = np.concatenate([states, 0.5 * np.ones_like(states)], axis=1)

        fig, axs = plt.subplots(2)

        t = _plot_constraint(axs[0], states, X, Y, N, get_plt_states, type="constraint")
        plt.colorbar(t)

        t = _plot_constraint(axs[1], states, X, Y, N, get_plt_states, type="feasibility")
        plt.colorbar(t)

        # Remove duplicate labels from legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        if save_path is not None:
            plt.savefig(save_path)

        if debug:
            plt.show()
        plt.clf()

    
if __name__ == "__main__":
    
    env = BallND(return_cost=False, dense_const=True)
    obs = env.reset()

    env.render()
    for i in range(1000):
        action = np.array([-0.1, -0.2])
        
        obs, step_reward, done, info = env.step(action)
        # print(info)
        env.render()
        if done:
            obs = env.reset()
        
        