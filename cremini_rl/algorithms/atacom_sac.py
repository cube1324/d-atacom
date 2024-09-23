import torch
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC
from mushroom_rl.utils.parameters import to_parameter

from cremini_rl.utils.null_space import batch_smooth_basis, smooth_basis
import numpy as np


class AtacomSACBaseline(SAC):
    def __init__(self, mdp_info, control_system, atacom_lam, atacom_beta, constraint_func, use_viability,
                 actor_mu_params,
                 actor_sigma_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, use_log_alpha_loss=False,
                 log_std_min=-20, log_std_max=2, target_entropy=None, critic_fit_params=None):
        super().__init__(mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, batch_size,
                         initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, use_log_alpha_loss,
                         log_std_min, log_std_max, target_entropy, critic_fit_params)

        self._control_system = control_system
        self._atacom_lam = to_parameter(atacom_lam)
        self._atacom_beta = to_parameter(atacom_beta)
        self._constraint_func = constraint_func

        self._add_save_attr(_control_system='mushroom',
                            _atacom_lam='mushroom',
                            _atacom_beta='mushroom')

        self._use_viability = use_viability
        self.derivation_step_size = 1e-4
        # self.K = 0.5


    def fit(self, dataset, **info):
        new_dataset = []
        for sample in dataset:
            new_dataset.append(sample[:4] + sample[5:])

        super().fit(new_dataset, **info)

    def J_slack(self, slack):
        out = np.zeros(slack.shape + (slack.shape[1],))
        np.einsum('ijj->ij', out)[:] = 1 / np.maximum(np.exp(-self._atacom_beta() * slack), 1e-5) - 1

        return out

    def preprocess_action(self, state, alpha, next_cost):
        # alpha = np.array([1, 1, 1])

        state_tensor = torch.from_numpy(state)
        q = self._control_system.get_q(state_tensor.numpy())
        
        # Only get classic constraint k(s), need to implement viability here
        cons, J_k, J_x, K = self._constraint_func(q)
        lam = self._atacom_lam()

        G = self._control_system.G(q)
        f = self._control_system.f(q)

        drift = np.zeros(cons.shape + (1,))

        if self._use_viability:
            # Assumes states is stacked as [q, q_dot]
            state_dim = G.shape[0] // 2
            # use lower half of the system as new dynamics
            G = G[state_dim:]
            f = f[:, state_dim:]

            indicator = K == 0
            # Only velocity jacobian for vel constraint
            J_k_velocity = J_k[:, indicator, state_dim:]
            # Only position jacobian for viability constraint
            J_k_viability = J_k[:, ~indicator, :state_dim]

            # Build new jacobian
            J_k = np.concatenate([J_k_viability, J_k_velocity], axis=1)

            q_dot = q[:, state_dim:, None]

            # Transform to viability constraint
            cons = cons + K * (J_k @ q_dot).squeeze(2)

            q_delta = q.copy()
            q_delta[:, :state_dim] += q_delta[:, state_dim:] * self.derivation_step_size
            _, J_k_delta, _, _ = self._constraint_func(q_delta)
            J_k_viability_delta = J_k_delta[:, ~indicator, :state_dim]
            J_k_dot = (J_k_viability_delta - J_k_viability) / self.derivation_step_size

            drift[:, ~indicator] += (J_k_viability + J_k_dot) @ q_dot

            # Estimate J_k_dot by finite difference
            # J_k_dot = np.zeros(J_k_viability.shape + (state_dim,))
            # for i in range(state_dim):
            #     q_delta = q.copy()
            #     q_delta[:, i] += self.derivation_step_size
            #     _, J_k_delta, _, _ = self._constraint_func(q_delta)
            #     J_k_viability_delta = J_k_delta[:, ~indicator, :state_dim]
            #     J_k_dot[:, :, :, i] = (J_k_viability_delta - J_k_viability) / self.derivation_step_size

            # drift[:, ~indicator] += (J_k_viability + (J_k_dot @ q_dot).squeeze(3)) @ q_dot

        slack = np.maximum(-cons, 1e-5)

        J_G = J_k @ G  # (B, k, u)

        J_u = np.concatenate((J_G, self.J_slack(slack)), axis=-1)  # (B, k, u + k)

        drift += J_k @ f  # (B, k)

        drift = np.maximum(drift, 0)

        B_u = batch_smooth_basis(J_u)

        c = cons + slack

        # J_u_inv = torch.linalg.pinv(J_u, atol=0.1)
        J_u_inv = np.linalg.pinv(J_u)

        drift_compensation = J_u_inv @ drift
        contraction_term = lam * J_u_inv @ c[:, :, None]

        b = - drift_compensation - contraction_term

        alpha = np.atleast_2d(np.clip(alpha, -1, 1))

        tangential_term = B_u @ alpha[:, :, None]

        u = tangential_term + b

        action = u[:, :alpha.shape[1]].flatten()

        # print(np.max(cons))
        # print(drift_compensation.flatten()[:3])
        # print(contraction_term.flatten()[:3])
        # print(tangential_term.flatten()[:3])
        # print(action)
        # print("-" * 10)

        return action
