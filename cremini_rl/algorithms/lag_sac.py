from mushroom_rl.algorithms.actor_critic.deep_actor_critic.sac import SAC, SACPolicy

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import to_parameter

from copy import deepcopy
from itertools import chain

from cremini_rl.utils.constraint_replay_memory import SafeReplayMemory
from cremini_rl.utils.plot import plot_zero_level


class LagSAC(SAC):
    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, constraint_params, lr_beta, damp_scale=0,
                 cost_limit=None, fixed_cost_penalty=None, cost_constraint=None, use_log_alpha_loss=False,
                 log_std_min=-20, log_std_max=2, target_entropy=None, critic_fit_params=None):
        """
        Constructor.

        Args:
            actor_mu_params (dict): parameters of the actor mean approximator to build;
            actor_sigma_params (dict): parameters of the actor sigma approximator to build;
            actor_optimizer (dict): parameters to specify the actor optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to build;
            batch_size ((int, Parameter)): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before starting the learning;
            max_replay_size (int): the maximum number of samples in the replay memory;
            warmup_transitions ([int, Parameter]): number of samples to accumulate in the replay memory to start the
                policy fitting;
            tau ([float, Parameter]): value of coefficient for soft updates;
            lr_alpha ([float, Parameter]): Learning rate for the entropy coefficient;
            use_log_alpha_loss (bool, False): whether to use the original implementation loss or the one from the
                paper;
            log_std_min ([float, Parameter]): Min value for the policy log std;
            log_std_max ([float, Parameter]): Max value for the policy log std;
            target_entropy (float, None): target entropy for the policy, if None a default value is computed;
            critic_fit_params (dict, None): parameters of the fitting algorithm of the critic approximator.

        """
        assert not (cost_limit is None and cost_constraint is None and fixed_cost_penalty is None)

        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)

        self._use_log_alpha_loss = use_log_alpha_loss

        self._damp_scale = damp_scale
        self._fixed_cost_penalty = fixed_cost_penalty
        self._cost_constraint = cost_constraint

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)
        else:
            self._target_entropy = target_entropy

        self._replay_memory = SafeReplayMemory(initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_constraint_params = deepcopy(constraint_params)

        self._constraint_approximator = Regressor(TorchApproximator, **constraint_params)
        self._target_constraint_approximator = Regressor(TorchApproximator, **target_constraint_params)

        self._init_target(self._constraint_approximator, self._target_constraint_approximator)

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator, **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator, **target_critic_params)

        actor_mu_approximator = Regressor(TorchApproximator, **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator, **actor_sigma_params)

        policy = SACPolicy(actor_mu_approximator, actor_sigma_approximator, mdp_info.action_space.low,
                           mdp_info.action_space.high, log_std_min, log_std_max)

        self._init_target(self._critic_approximator, self._target_critic_approximator)

        self._log_alpha = torch.tensor(0., dtype=torch.float32)

        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        if self._fixed_cost_penalty is None:
            self._beta = torch.tensor(0., dtype=torch.float32)

            if policy.use_cuda:
                self._beta = self._beta.cuda().requires_grad_()
            else:
                self._beta.requires_grad_()

            self._beta_optim = optim.Adam([self._beta], lr=lr_beta)

            self._add_save_attr(_beta_optim='torch')

        else:
            self._beta = torch.tensor(fixed_cost_penalty)

        self.training_loss = []

        policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                  actor_sigma_approximator.model.network.parameters())

        super(SAC, self).__init__(mdp_info, policy, actor_optimizer, policy_parameters)

        if self._cost_constraint is None and cost_limit is not None:
            self._cost_constraint = (cost_limit * (1 - self.mdp_info.gamma ** self.mdp_info.horizon) /
                                     (1 - self.mdp_info.gamma) / self.mdp_info.horizon)

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _use_log_alpha_loss='primitive',
            _log_alpha='torch',
            _alpha_optim='torch',
            _damp_scale='primitive',
            _fixed_cost_penalty='primitive',
            _cost_constraint='primitive',
            _constraint_approximator='mushroom',
            _target_constraint_approximator='mushroom',
            _beta='torch',
        )

    def _loss(self, state, action_new, log_prob, cost):
        q_0 = self._critic_approximator(state, action_new, output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new, output_tensor=True, idx=1)

        c = self._constraint_approximator(state, action_new, output_tensor=True)

        q = torch.min(q_0, q_1)

        damp = self._damp_scale * torch.mean(self._cost_constraint - cost)

        return (self._alpha * log_prob - q + (self.beta.detach() - damp) * c).mean()

    def fit(self, dataset, **info):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, cost, absorbing, _ = self._replay_memory.get(self._batch_size())
            
            cost = np.maximum(cost, 0)

            current_cost = self._constraint_approximator(state, action, output_tensor=True)
            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)

                loss = self._loss(state, action_new, log_prob, current_cost)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob.detach())
                self._update_beta(current_cost.detach())

            q_next, c_next = self._next_q_and_c(next_state, absorbing)

            q = reward + self.mdp_info.gamma * q_next
            c = cost + self.mdp_info.gamma * c_next

            self._critic_approximator.fit(state, action, q, **self._critic_fit_params)
            self._constraint_approximator.fit(state, action, c)

            self.training_loss.append([self._constraint_approximator.model._last_loss, 0])

            self._update_target(self._critic_approximator, self._target_critic_approximator)
            self._update_target(self._constraint_approximator, self._target_constraint_approximator)

    def _next_q_and_c(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the action returned by the actor.

        """
        a, log_prob_next = self.policy.compute_action_and_log_prob(next_state)

        q = self._target_critic_approximator.predict(
            next_state, a, prediction='min') - self._alpha_np * log_prob_next
        q *= 1 - absorbing

        c = self._target_constraint_approximator.predict(next_state, a)
        c *= 1 - absorbing

        return q, c

    def _update_beta(self, cost):
        if self._fixed_cost_penalty is None:

            beta_loss = torch.mean(self.beta * (self._cost_constraint - cost))
            self._beta_optim.zero_grad()
            beta_loss.backward()
            self._beta_optim.step()

    @property
    def beta(self):
        return F.softplus(self._beta)

    def plot_constraint(self, ax, states, X, Y, N, get_plt_states, type=None):

        actions = self.policy.draw_action(states)
        # actions = np.random.uniform(-1, 1, (states.shape[0], 2))

        constraint = self._constraint_approximator.predict(states, actions)

        constraint_square = constraint.reshape(N, N)

        plot_zero_level(ax, X, Y, constraint_square, color="tab:red", label="constraint")

        t = ax.scatter(*get_plt_states(states), c=constraint, vmin=-1, vmax=1)

        state, action, reward, next_state, cost, absorbing, _ = self._replay_memory.get(self._batch_size())

        ax.scatter(*state[:, :2].T, c="white", s=10)

        return t