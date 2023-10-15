import numpy as np

import torch
import torch.nn.functional as F

from cremini_rl.algorithms.lag_sac import LagSAC
from cremini_rl.utils.plot import plot_zero_level
from scipy.stats import norm


class WCSAC(LagSAC):
    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, constraint_params, lr_beta,
                 accepted_risk, damp_scale=0, constraint_type="quantile",
                 cost_limit=None, fixed_cost_penalty=None, cost_constraint=None, use_log_alpha_loss=False,
                 log_std_min=-20, log_std_max=2, target_entropy=None, critic_fit_params=None):

        self._accepted_risk = accepted_risk
        self._margin = norm.pdf(norm.ppf(1 - accepted_risk)) / accepted_risk
        self._cbf_gamma = mdp_info.gamma
        self._constraint_type = constraint_type

        if self._constraint_type == "gaussian":
            constraint_params["loss"] = self.gaussian_wasserstein_dist
            constraint_params["n_fit_targets"] = 4
        elif self._constraint_type == "quantile":
            constraint_params["loss"] = self.quantile_huber_loss
            constraint_params["n_fit_targets"] = 2
            self._n_taus = 8

        self._add_save_attr(_margin='primitive',
                            _cbf_gamma='primitive',
                            _accepted_risk='primitive',
                            _constraint_type='primitive')

        super(WCSAC, self).__init__(mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, constraint_params, lr_beta,
                 damp_scale,
                 cost_limit, fixed_cost_penalty, cost_constraint, use_log_alpha_loss,
                 log_std_min, log_std_max, target_entropy, critic_fit_params)

    def _get_cvar(self, state, action):
        if self._constraint_type == "gaussian":
            mean, std = self._constraint_approximator(state, action, output_tensor=True)
            return mean + std * self._margin

        elif self._constraint_type == "quantile":
            # Approximate Cvar from samples
            tau, _ = self.get_tau(len(state), self._accepted_risk)

            cvar = self._constraint_approximator(state, action, tau, output_tensor=True)

            return cvar.mean(axis=1).squeeze()



    def _loss(self, state, action_new, log_prob, actual_margin):
        q_0 = self._critic_approximator(state, action_new, output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new, output_tensor=True, idx=1)

        # Cvar of actor action
        actor_margin = self._get_cvar(state, action_new)

        q = torch.min(q_0, q_1)

        damp = self._damp_scale * torch.mean(self._cost_constraint - actual_margin)

        return (self._alpha * log_prob - q + (self.beta.detach() - damp) * actor_margin).mean()


    def fit(self, dataset, **info):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, cost, absorbing, _ = self._replay_memory.get(self._batch_size())
            
            cost = np.maximum(cost, 0)

            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)

                current_margin = self._get_cvar(state, action)

                loss = self._loss(state, action_new, log_prob, current_margin)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob.detach())
                self._update_beta(current_margin.detach())

            next_a, log_prob_next = self.policy.compute_action_and_log_prob(next_state)

            next_q = self._target_critic_approximator.predict(
                next_state, next_a, prediction='min') - self._alpha_np * log_prob_next
            next_q *= 1 - absorbing

            q = reward + self.mdp_info.gamma * next_q

            self._critic_approximator.fit(state, action, q, **self._critic_fit_params)

            self.fit_constraint(state, action, cost, next_state, next_a, absorbing)

            self._update_target(self._critic_approximator, self._target_critic_approximator)
            self._update_target(self._constraint_approximator, self._target_constraint_approximator)

    def fit_constraint(self, state, action, cost, next_state, next_a, absorbing):

        if self._constraint_type == "gaussian":
            c_next = self._target_constraint_approximator.predict(next_state, next_a)

            self._constraint_approximator.fit(state, action, cost, c_next, absorbing,
                                              np.ones_like(cost) * self._cbf_gamma)

        elif self._constraint_type == "quantile":
            tau, tau_prime = self.get_tau(self._batch_size())

            c_next = self._target_constraint_approximator.predict(next_state, next_a, tau_prime)

            target = np.repeat(cost.reshape(-1, 1, 1), self._n_taus, axis=1) + self._cbf_gamma * c_next
            target *= 1 - np.repeat(absorbing.reshape(-1, 1, 1), self._n_taus, axis=1)

            self._constraint_approximator.fit(state, action, tau, target, tau)

        self.training_loss.append([0, self._constraint_approximator.model._last_loss])


    def get_tau(self, batch_size, cvar = 1):
        presum_tau = np.random.rand(batch_size, self._n_taus) + 0.1
        presum_tau /= presum_tau.sum(axis=-1, keepdims=True)

        tau = np.cumsum(presum_tau, axis=-1)

        tau_hat = np.zeros_like(tau)
        tau_hat[:, 0:1] = tau[:, 0:1] / 2
        tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.

        # get values for cvar
        tau = 1 - cvar + cvar * tau
        tau_prime = 1 - cvar + cvar * tau_hat

        return tau, tau_prime

    @staticmethod
    def gaussian_wasserstein_dist(predicted_cost, cost, next_cost, absorbing, gamma):
        next_mu = next_cost[0]
        next_sigma = torch.clamp(next_cost[1]**2, 1e-8, 1e8)

        predicted_mu = predicted_cost[0]
        predicted_mu_detach = predicted_mu.detach()
        predicted_sigma = torch.clamp(predicted_cost[1].flatten()**2, 1e-8, 1e8)

        mu_hat = cost + (1 - absorbing) * gamma * next_mu
        J_mu = F.mse_loss(predicted_mu, mu_hat)


        sigma_hat = cost ** 2 \
                    + 2 * cost * gamma * next_cost[0] \
                    + gamma ** 2 * (next_sigma + next_cost[0] ** 2) \
                    - predicted_mu_detach ** 2

        sigma_hat = torch.clamp(sigma_hat, 1e-8, 1e8)
        J_sigma = torch.mean(sigma_hat + predicted_sigma - 2 * torch.sqrt(sigma_hat * predicted_sigma))

        return J_mu + J_sigma

    @staticmethod
    def quantile_huber_loss(y_hat, target, tau):
        batch_size = y_hat.size(0)
        N = y_hat.size(1)
        N_prime = target.size(1)

        tau = tau.repeat(1, N_prime)

        err = (target[:, None, :] - y_hat[:, :, None]).view(batch_size, N * N_prime)
        k = 0.01
        piece_idx = torch.abs(err) < k
        huber_loss = err ** 2 / 2. * piece_idx + k * (torch.abs(err) - k / 2) * (1 - piece_idx.float())
        loss = huber_loss / k * torch.abs(tau - (err < 0).float())
        return loss.mean()

    def plot_constraint(self, ax, states, X, Y, N, get_plt_states, type=None):

        actions = self.policy.draw_action(states)
        # actions = np.random.uniform(-1, 1, (states.shape[0], 2))

        # mean, std = self._target_constraint_approximator.predict(states, actions)
        # margin = mean + std * self._margin
        with torch.no_grad():
            mean = self._get_cvar(states, actions)

        constraint_square = mean.reshape(N, N)
        # margin_square = margin.reshape(N, N)

        plot_zero_level(ax, X, Y, constraint_square, color="tab:red", label="constraint")
        # plot_zero_level(ax, X, Y, margin_square, color="white", label="margin")

        t = ax.scatter(*get_plt_states(states), c=mean, vmin=-1, vmax=1)

        # state, action, reward, next_state, cost, absorbing, _ = self._replay_memory.get(self._batch_size())
        #
        # ax.scatter(*next_state[:, :2].T, c="r", s=10, vmin=-1, vmax=1)

        return t