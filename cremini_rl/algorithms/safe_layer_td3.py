import torch

import torch.nn.functional as F
from torch import nn, optim

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import Parameter, to_parameter

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DDPG

from copy import deepcopy
import numpy as np


from cremini_rl.utils.constraint_replay_memory import SafeLayerReplayMemory


class SafeLayerDDPG(DDPG):
    def __init__(self, mdp_info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau, delta, constraint_params, warmup_transitions,
                 policy_delay=1, critic_fit_params=None, actor_predict_params=None, critic_predict_params=None):

        self._constraint_approximator = Regressor(TorchApproximator, **constraint_params)

        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params
        self._actor_predict_params = dict() if actor_predict_params is None else actor_predict_params
        self._critic_predict_params = dict() if critic_predict_params is None else critic_predict_params

        self._batch_size = to_parameter(batch_size)
        self._tau = to_parameter(tau)
        self._delta = to_parameter(delta)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._policy_delay = to_parameter(policy_delay)
        self._fit_count = 0

        self._replay_memory = SafeLayerReplayMemory(initial_replay_size, max_replay_size)

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(TorchApproximator,
                                             **actor_params)
        self._target_actor_approximator = Regressor(TorchApproximator,
                                                    **target_actor_params)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)
        self._init_target(self._actor_approximator,
                          self._target_actor_approximator)

        policy = policy_class(self._actor_approximator, **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        self._add_save_attr(
            _critic_fit_params='pickle',
            _critic_predict_params='pickle',
            _actor_predict_params='pickle',
            _batch_size='mushroom',
            _tau='mushroom',
            _delta='mushroom',
            _policy_delay='mushroom',
            _fit_count='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _target_actor_approximator='mushroom',
            _constraint_approximator='mushroom'
        )

        super(DDPG, self).__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset, **info):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, cost, prev_cost, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            # Learn Cost
            self._constraint_approximator.fit(state, action, prev_cost, cost)

            if self._replay_memory.size > self._warmup_transitions():
                # DDPG

                q_next = self._next_q(next_state, absorbing)
                q = reward + self.mdp_info.gamma * q_next

                self._critic_approximator.fit(state, action, q,
                                            **self._critic_fit_params)

                if self._fit_count % self._policy_delay() == 0:
                    loss = self._loss(state)
                    self._optimize_actor_parameters(loss)

                self._update_target(self._critic_approximator,
                                    self._target_critic_approximator)
                self._update_target(self._actor_approximator,
                                    self._target_actor_approximator)

                self._fit_count += 1

    def preprocess_action(self, state, action, cost):
        g = self._constraint_approximator.predict(state)
        delta_c = g @ action
        expected_c = delta_c + cost

        if expected_c < self._delta():
            return action

        # Equation 5
        lam = np.maximum((expected_c - self._delta()) / (g @ g.T + 1e-8), 0)

        # Equation 6
        new_action = action - lam * g

        return new_action.flatten()

    @staticmethod
    def safelayer_loss(g, action, prev_c, target_c):
        delta_c = g.unsqueeze(1) @ action.unsqueeze(2)
        return F.mse_loss(delta_c.squeeze() + prev_c, target_c)


class SafeLayerTD3(SafeLayerDDPG):
    def __init__(self, mdp_info, policy_class, policy_params, actor_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau, delta, constraint_params, warmup_transitions,
                 policy_delay=2, noise_std=.2, noise_clip=.5, critic_fit_params=None):
        """
        Constructor.

        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            actor_params (dict): parameters of the actor approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ([int, Parameter]): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau ([float, Parameter]): value of coefficient for soft updates;
            policy_delay ([int, Parameter], 2): the number of updates of the critic after
                which an actor update is implemented;
            noise_std ([float, Parameter], .2): standard deviation of the noise used for
                policy smoothing;
            noise_clip ([float, Parameter], .5): maximum absolute value for policy smoothing
                noise;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._noise_std = to_parameter(noise_std)
        self._noise_clip = to_parameter(noise_clip)

        if 'n_models' in critic_params.keys():
            assert(critic_params['n_models'] >= 2)
        else:
            critic_params['n_models'] = 2

        self._add_save_attr(
            _noise_std='mushroom',
            _noise_clip='mushroom'
        )

        super().__init__(mdp_info, policy_class, policy_params,  actor_params,
                         actor_optimizer, critic_params, batch_size,
                         initial_replay_size, max_replay_size, tau, delta, constraint_params,  warmup_transitions, policy_delay,
                         critic_fit_params)

    def _loss(self, state):
        action = self._actor_approximator(state, output_tensor=True, **self._actor_predict_params)
        q = self._critic_approximator(state, action, idx=0, output_tensor=True, **self._critic_predict_params)

        return -q.mean()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a = self._target_actor_approximator(next_state, **self._actor_predict_params)

        low = self.mdp_info.action_space.low
        high = self.mdp_info.action_space.high
        eps = np.random.normal(scale=self._noise_std(), size=a.shape)
        eps_clipped = np.clip(eps, -self._noise_clip(), self._noise_clip.get_value())
        a_smoothed = np.clip(a + eps_clipped, low, high)

        q = self._target_critic_approximator.predict(next_state, a_smoothed,
                                                     prediction='min', **self._critic_predict_params)
        q *= 1 - absorbing

        return q