import numpy as np
from copy import deepcopy
from itertools import chain

import torch
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.utils.replay_memory import ReplayMemory

from cremini_rl.utils.constraint_replay_memory import SafeReplayMemory
from cremini_rl.algorithms.gaussian_atacom_sac import AtacomSACPolicy, GaussianAtacomSAC


class AtacomIQNSACPolicy(AtacomSACPolicy):
    def __init__(self, mu_approximator, sigma_approximator, constraint_approximator, control_system, mdp_info,
                 accepted_risk, delta, atacom_lam, atacom_beta, target_entropy, min_a, max_a, log_std_min, log_std_max):
        self._risk_logit = torch.tensor(np.log(accepted_risk / (1 - accepted_risk))).float()

        super().__init__(mu_approximator, sigma_approximator, constraint_approximator, control_system, mdp_info,
                         accepted_risk, delta, atacom_lam, atacom_beta, target_entropy, min_a, max_a, log_std_min,
                         log_std_max)

        self._add_save_attr(
            _risk_logit='torch',
        )

    def compute_constraint(self, q, x):
        q_tensor = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        state = torch.hstack([q_tensor, x_tensor])

        def dummy_aux_fun(x):
            out = self.compute_constraint(x)
            return out, out.detach()

        def get_vjp(x):
            return torch.func.jacrev(dummy_aux_fun, argnums=0, has_aux=True)(x)

        J, cons = torch.vmap(get_vjp)(state)

        J_q = J[:, :, :self._control_system.dim_q].detach()
        J_x = J[:, :, self._control_system.dim_q:].detach()

        return cons.double(), J_q.double(), J_x.double()

    def compute_constraint(self, state):
        delta = self._delta().detach().to(self.device)
        tau = (torch.ones(1) - self.accepted_risk()).to(self.device)
        cbf_value = self._constraint_approximator.model.network(state, tau)
        return cbf_value - delta

    def accepted_risk(self):
        return torch.sigmoid(self._risk_logit)


class IQNAtacomSAC(GaussianAtacomSAC):
    def __init__(self, mdp_info, control_system, accepted_risk, actor_mu_params, actor_sigma_params, actor_optimizer,
                 critic_params, batch_size, initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha,
                 cost_budget, constraint_params, atacom_lam, atacom_beta,
                 lr_delta, init_delta, delta_warmup_transitions,
                 num_quantile_samples, num_next_quantile_samples,
                 use_log_alpha_loss=False, log_std_min=-20, log_std_max=2, target_entropy=None,
                 critic_fit_params=None):
        """
        Constructor.

        Args:
            mdp_info (MdpInfo): information about the MDP;
            control_system (ControlSystem): control system to use;
            accepted_risk (float, None): accepted risk for the CBF, if None the risk is updated based on the cost;
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

        constraint_params['loss'] = self.quantile_huber_loss2

        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)

        self._use_log_alpha_loss = use_log_alpha_loss

        # Maximum acceptable sum of cost per episode
        self._cost_budget = cost_budget

        # Sum of cost of current episode
        self._episode_step_count = 0
        self._episode_costs = []
        self._episode_states = []

        self._episode_end = False

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)
        else:
            self._target_entropy = target_entropy

        self._replay_memory = SafeReplayMemory(initial_replay_size, max_replay_size)

        self._violation_replay_memory = ReplayMemory(0, max_replay_size // 10)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator, **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator, **target_critic_params)

        target_constraint_params = deepcopy(constraint_params)
        self._constraint_approximator = Regressor(TorchApproximator, **constraint_params)
        self._target_constraint_approximator = Regressor(TorchApproximator, **target_constraint_params)

        actor_mu_approximator = Regressor(TorchApproximator, **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator, **actor_sigma_params)

        self._init_target(self._critic_approximator, self._target_critic_approximator)
        self._init_target(self._constraint_approximator, self._target_constraint_approximator)

        policy = AtacomIQNSACPolicy(actor_mu_approximator, actor_sigma_approximator,
                                    self._constraint_approximator, control_system, mdp_info, accepted_risk, self.delta,
                                    atacom_lam,
                                    atacom_beta, self._target_entropy, mdp_info.action_space.low,
                                    mdp_info.action_space.high,
                                    log_std_min, log_std_max)

        self._log_alpha = torch.tensor(0., dtype=torch.float32)
        self._delta_value = torch.tensor(np.maximum(init_delta, 0.1), dtype=torch.float32)

        self._log_alpha.requires_grad_()
        self._delta_value.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)
        self._delta_optim = optim.Adam([self._delta_value], lr=lr_delta)
        self._delta_warmup_transitions = delta_warmup_transitions

        policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                  actor_sigma_approximator.model.network.parameters())

        self._num_quantile_samples = num_quantile_samples
        self._num_next_quantile_samples = num_next_quantile_samples

        self.training_loss = []
        self.cbf_reg_loss = []

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _cost_budget='primitive',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _violation_replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _constraint_approximator='mushroom',
            _target_constraint_approximator='mushroom',
            _use_log_alpha_loss='primitive',
            _log_alpha='torch',
            _log_delta='torch',
            _alpha_optim='torch',
            _delta_optim='torch',
            _delta_warmup_transitions='primitive',
            _num_quantile_samples='primitive',
            _num_next_quantile_samples='primitive',
        )

        super(GaussianAtacomSAC, self).__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset, **info):
        self._add_episode_cost(dataset)

        if dataset[-1][-1] and self._replay_memory.size > self._delta_warmup_transitions:
            self.update_delta()

        self._replay_memory.add(dataset)

        self._violation_replay_memory.add(self._get_violations(dataset))

        if self._replay_memory.initialized:
            state, action, reward, next_state, cost, absorbing, _ = self._replay_memory.get(self._batch_size())

            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob_new = self.policy.compute_action_and_log_prob_t(state)
                loss = self._loss(state, action_new, log_prob_new)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob_new.detach())

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q, **self._critic_fit_params)

            # Fit CBF
            if self._violation_replay_memory.size > self._batch_size() // 10:
                violating_state, _, violating_cost, violating_next_state, violating_absorbing, _ = self._violation_replay_memory.get(
                    self._batch_size() // 10)

                state = np.concatenate([state, violating_state], axis=0)
                cost = np.concatenate([cost, violating_cost], axis=0)
                next_state = np.concatenate([next_state, violating_next_state], axis=0)
                absorbing = np.concatenate([absorbing, violating_absorbing], axis=0)

            constraint_state, next_constraint_state = self.to_constraint_state(state, next_state)

            with torch.no_grad():
                next_quantile = torch.rand(next_constraint_state.shape[0], self._num_next_quantile_samples)
                next_value = self._target_constraint_approximator.predict(
                    next_constraint_state, next_quantile, output_tensor=True)

            error = torch.tensor(np.maximum(cost, 0), dtype=torch.float32, device=self.device)
            target_value = error[:, None] + torch.tensor(1 - absorbing, device=self.device)[:, None] * self.mdp_info.gamma * next_value

            self._constraint_approximator.model._optimizer.zero_grad()

            quantile = torch.rand(constraint_state.shape[0], self._num_next_quantile_samples, device=self.device)
            value = self._constraint_approximator.predict(constraint_state, quantile, output_tensor=True)

            loss = self.quantile_huber_loss2(value, target_value, quantile, reduction='none')

            loss = loss.mean()

            loss.backward()

            self._constraint_approximator.model._optimizer.step()

            self.training_loss.append(loss.detach().cpu().item())
            # self.cbf_reg_loss.append(reg_loss.detach().item())

            self._update_target(self._constraint_approximator, self._target_constraint_approximator)
            self._update_target(self._critic_approximator, self._target_critic_approximator)

    def _post_load(self):
        super()._post_load()
        self.policy._constraint_approximator = self._constraint_approximator

    @staticmethod
    def quantile_huber_loss2(q, q_target, tau, reduction='mean'):
        q = q.unsqueeze(-1)
        q_target = q_target.unsqueeze(-2)
        tau = tau.unsqueeze(-1)

        q_expanded, q_target_expanded = torch.broadcast_tensors(q, q_target)
        sign = ((q_target_expanded - q_expanded) < 0).float()
        L_delta = smooth_l1_loss(q_target_expanded, q_expanded, reduction='none')
        L = torch.abs(tau - sign) * L_delta
        if reduction == 'mean':
            return L.sum(dim=-1).mean()
        return L.sum(dim=-1).mean(dim=1)

    @staticmethod
    def quantile_huber_loss(y_hat, target, tau, reduction='mean'):
        batch_size = y_hat.size(0)
        N = y_hat.size(1)
        N_prime = target.size(1)

        tau = tau.repeat(1, N_prime)

        err = (target[:, None, :] - y_hat[:, :, None]).view(batch_size, N * N_prime)
        k = 0.01
        piece_idx = torch.abs(err) < k
        huber_loss = err ** 2 / 2. * piece_idx + k * (torch.abs(err) - k / 2) * (1 - piece_idx.float())
        loss = huber_loss / k * torch.abs(tau - (err < 0).float())

        loss = loss.mean(dim=1)

        if reduction == 'mean':
            return loss.mean()
        else:
            return loss
