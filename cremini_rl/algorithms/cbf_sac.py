import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.utils.replay_memory import ReplayMemory

from copy import deepcopy
from itertools import chain

from cremini_rl.utils.constraint_replay_memory import SafeReplayMemory

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class CBFSACPolicy(Policy):
    """
    Class used to implement the policy used by the Soft Actor-Critic algorithm.
    The policy is a Gaussian policy squashed by a tanh. This class implements the compute_action_and_log_prob and the
    compute_action_and_log_prob_t methods, that are fundamental for the internals calculations of the SAC algorithm.

    """

    def __init__(self, mu_approximator, sigma_approximator, constraint_approximator, control_system, target_entropy,
                 min_a, max_a, log_std_min, log_std_max,
                 analytical_const=None):
        """
        Constructor.

        Args:
            mu_approximator (Regressor): a regressor computing mean in given a state;
            sigma_approximator (Regressor): a regressor computing the variance in given a state;
            min_a (np.ndarray): a vector specifying the minimum action value for each component;
            max_a (np.ndarray): a vector specifying the maximum action value for each component.
            log_std_min ([float, Parameter]): min value for the policy log std;
            log_std_max ([float, Parameter]): max value for the policy log std.

        """

        if mu_approximator.model.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self._mu_approximator = mu_approximator
        self._sigma_approximator = sigma_approximator

        self.min_a = min_a
        self.max_a = max_a

        self._delta_a = to_float_tensor(.5 * (max_a - min_a)).to(self.device)
        self._central_a = to_float_tensor(.5 * (max_a + min_a)).to(self.device)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._numeric_eps = 1e-6

        # Dynamics system of the agent
        self._control_system = control_system

        # Action space used for rescaling the action after atacom transformation
        # self._action_low = torch.from_numpy(mdp_info.action_space.low)
        # self._action_high = torch.from_numpy(mdp_info.action_space.high)

        # Assumes that reasonable cost is roughly capped at 1
        # self._cbf_scale = 1 / (1 - mdp_info.gamma)

        # Neural Networks for the CBF
        self._constraint_approximator = constraint_approximator

        # # ATACOM gain for error correction
        # self._atacom_lam = to_parameter(atacom_lam)
        # # ATACOM shape of slack function (how close to the constraint the agent is allowed)
        # self._atacom_beta = to_parameter(atacom_beta)

        # Cvar margin
        # self._margin = norm.pdf(norm.ppf(1 - accepted_risk)) / accepted_risk
        #
        # self._delta = delta

        self._target_entropy = torch.tensor(target_entropy)

        self._debug_residual = []

        self._learn_constraint = False
        self._learn_cbf = True

        self._analytical_const = analytical_const
        self.K = 0.8
        self.derivation_step_size = 1e-4

        self._add_save_attr(
            _mu_approximator='mushroom',
            _sigma_approximator='mushroom',
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            _numeric_eps='primitive',
            _control_system='mushroom',
            # _cbf_scale='primitive',
            _atacom_lam='mushroom',
            _atacom_beta='mushroom',
            _margin='primitive',
            min_a='numpy',
            max_a='numpy',
        )
        self.build_qp()

    def build_qp(self):
        self.clf_lambda = 1.0

        u = cp.Variable(self._control_system.dim_u)
        clf_relaxation = cp.Variable(1, nonneg=True)

        V_param = cp.Parameter(1, nonneg=True)
        L_fV_param = cp.Parameter(1)
        L_GV_param = cp.Parameter((self._control_system.dim_u))

        clf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)
        u_ref_param = cp.Parameter(self._control_system.dim_u)

        constraints = []
        constraints.append(
            L_fV_param
            + L_GV_param @ u
            + self.clf_lambda * V_param
            - clf_relaxation
            <= 0)

        for control_idx in range(self._control_system.dim_u):
            constraints.append(u[control_idx] >= self.min_a[control_idx])
            constraints.append(u[control_idx] <= self.max_a[control_idx])

        objective_expression = cp.sum_squares(u - u_ref_param)
        objective_expression += cp.multiply(clf_relaxation_penalty_param, clf_relaxation)
        objective = cp.Minimize(objective_expression)

        # Finally, create the optimization problem
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        variables = [u, clf_relaxation]
        parameters = [L_fV_param, L_GV_param]
        parameters += [V_param, u_ref_param, clf_relaxation_penalty_param]
        self.differentiable_qp_solver = CvxpyLayer(
            problem, variables=variables, parameters=parameters
        )

    def draw_action(self, state):
        action = self.compute_action_and_log_prob_t(np.atleast_2d(state), return_log_prob=False)
        return action.detach().cpu().numpy().squeeze()

    def convert_action(self, alpha, state, return_v=False, need_grad=True):
        q = self._control_system.get_q(state)
        q_tensor = torch.tensor(q, dtype=torch.float32, device=self.device, requires_grad=True)

        temp = self.solve_qp(alpha, q_tensor, return_v, need_grad)
        return temp

    def solve_qp(self, u_ref, q_tensor, return_v=False, need_grad=True):
        v, L_fV, L_GV = self._compute_V_and_model(q_tensor, need_grad)

        relaxation_penalty = 1e3

        # The differentiable solver must allow relaxation
        relaxation_penalty = min(relaxation_penalty, 1e6)

        # Assemble list of params
        params = []
        params.append(L_fV.squeeze(1))
        params.append(L_GV.squeeze(1))
        params.append(v)
        params.append(u_ref)
        params.append(torch.tensor([relaxation_penalty], dtype=torch.float32, device=self.device))

        # We've already created a parameterized QP solver, so we can use that
        result = self.differentiable_qp_solver(
            *params,
            solver_args={"max_iters": 1000},
        )

        # Extract the results
        u_result = result[0]
        r_result = result[1]

        if r_result.shape[0] == 1:
            self._debug_residual.append(r_result.detach().cpu().item())

        # if v > 0:
        #     print("-" * 50)
        #     print(v.item())
        #     print(u_result.detach().numpy())
        #     print(u_ref.detach().numpy())
        #     print(r_result.detach().numpy())

        if return_v:
            return u_result, r_result, v, L_fV, L_GV

        return u_result, r_result

    def get_V_and_jac(self, q_tensor, need_grad=True):
        pred = self._constraint_approximator.predict(q_tensor, output_tensor=True)

        # not sure if grad is needed
        jac = torch.autograd.grad(pred, q_tensor, torch.ones_like(pred), create_graph=need_grad)[0]

        return pred.reshape(-1, 1), jac[:, None, :]

    def _compute_V_and_model(self, q_tensor, need_grad=True):

        v, jac = self.get_V_and_jac(q_tensor, need_grad)

        f = torch.from_numpy(self._control_system.f(q_tensor.detach().numpy())).float().to(self.device)
        G = torch.from_numpy(self._control_system.G(q_tensor.detach().numpy())[None, :, :]).float().to(self.device)

        L_f = jac @ f
        L_G = jac @ G

        return v, L_f, L_G

    def compute_action_and_log_prob(self, state):
        """
        Function that samples actions using the reparametrization trick and the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        a, log_prob = self.compute_action_and_log_prob_t(state)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, return_log_prob=True, return_auxiliary=False):
        """
        Function that samples actions using the reparametrization trick and, optionally, the log probability for such
        actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log  probability or not.

        Returns:
            The actions sampled and, optionally, the log probability as torch tensors.

        """
        dist = self.distribution(state)
        a_raw = dist.rsample()
        a = torch.tanh(a_raw)
        action = a * self._delta_a + self._central_a

        if return_log_prob:
            log_prob = dist.log_prob(a_raw).sum(dim=1)
            log_prob -= torch.log(1. - a.pow(2) + self._numeric_eps).sum(dim=1)

            # _, N, M = B.size()
            # K = min(N, M)
            #
            # # Add delta for numerical stability
            # delta = torch.eye(K) * 1e-4
            # log_det_B = torch.logdet(B[:, :K, :K] + delta.to(B))
            #
            # log_prob -= torch.maximum(log_det_B, self._target_entropy)

            return action, log_prob
        return action

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is computed.

        Returns:
            The torch distribution for the provided states.

        """
        mu = self._mu_approximator.predict(state, output_tensor=True)
        log_sigma = self._sigma_approximator.predict(state, output_tensor=True)
        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())
        return torch.distributions.Normal(mu, log_sigma.exp())

    def entropy(self, state=None):
        """
        Compute the entropy of the policy.

        Args:
            state (np.ndarray): the set of states to consider.

        Returns:
            The value of the entropy of the policy.

        """
        _, log_pi = self.compute_action_and_log_prob(state)
        return -log_pi.mean()

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by the policy.

        """
        mu_weights = weights[:self._mu_approximator.weights_size]
        sigma_weights = weights[self._mu_approximator.weights_size:]

        self._mu_approximator.set_weights(mu_weights)
        self._sigma_approximator.set_weights(sigma_weights)

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        mu_weights = self._mu_approximator.get_weights()
        sigma_weights = self._sigma_approximator.get_weights()

        return np.concatenate([mu_weights, sigma_weights])

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._mu_approximator.model.use_cuda

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch optimizers.

        Returns:
            List of parameters to be optimized.

        """
        return chain(self._mu_approximator.model.network.parameters(),
                     self._sigma_approximator.model.network.parameters())


class CBFSAC(DeepAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    """

    def __init__(self, mdp_info, control_system, actor_mu_params, actor_sigma_params, actor_optimizer,
                 critic_params, batch_size, initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha,
                 constraint_params,
                 use_log_alpha_loss=False, log_std_min=-20, log_std_max=2, target_entropy=None, critic_fit_params=None):
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
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)

        self._use_log_alpha_loss = use_log_alpha_loss

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

        # self.lr_schedueler = optim.lr_scheduler.LambdaLR(self._constraint_approximator.model._optimizer,
        #                                                  lambda step: 0.97 ** (step // 10000))
        actor_mu_approximator = Regressor(TorchApproximator, **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator, **actor_sigma_params)

        self._init_target(self._critic_approximator, self._target_critic_approximator)
        self._init_target(self._constraint_approximator, self._target_constraint_approximator)

        policy = CBFSACPolicy(actor_mu_approximator, actor_sigma_approximator,
                              self._constraint_approximator, control_system, self._target_entropy,
                              mdp_info.action_space.low,
                              mdp_info.action_space.high,
                              log_std_min, log_std_max)

        self._log_alpha = torch.tensor(0., dtype=torch.float32)
        # self._delta_value = torch.tensor(torch.log(torch.expm1(init_delta)), dtype=torch.float32)

        # x = torch.tensor(1.87 * policy._margin, dtype=torch.float32)
        # # inverse of softplus
        # accepted_risk_equalizer = x + torch.log(-torch.expm1(-x))

        # self._delta_value += accepted_risk_equalizer

        self.epsilon = 0.01
        self.safe_level = 1

        self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                  actor_sigma_approximator.model.network.parameters())

        self.training_loss = []
        self.cbf_reg_loss = []

        # torch.autograd.set_detect_anomaly(True)

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _violation_replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _constraint_approximator='mushroom',
            _target_constraint_approximator='mushroom',
            _use_log_alpha_loss='primitive',
            _log_alpha='torch',
            _delta_value='torch',
            _alpha_optim='torch',
            _delta_optim='torch',
            _delta_warmup_transitions='primitive',
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def preprocess_action(self, state, action, cost):
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)

        u_result, residual = self.policy.convert_action(action_tensor, state, need_grad=False)

        # print(np.linalg.norm(action - u_result.detach().cpu().numpy().squeeze()))
        return u_result.detach().cpu().numpy().squeeze()
        # return action

    def fit(self, dataset, **info):
        self._replay_memory.add(dataset)

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

            q = self.policy._control_system.get_q(state)

            self._constraint_approximator.model._optimizer.zero_grad()

            # q_tensor = torch.tensor(q, dtype=torch.float32, device=self.device, requires_grad=True)
            # u_result, r_result, v, L_fV, L_GV = self.policy.solve_qp(q_tensor, return_v=True)
            # TODO no idea if i should use alpha or action here.
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)

            u_result, residual, v, L_fV, L_GV = self.policy.convert_action(action_tensor, state, return_v=True)

            # TODO no idea if i should use action / u_result or action here.
            loss = self.clbf_loss(v, cost, u_result, residual, L_fV, L_GV)

            loss.backward()

            self._constraint_approximator.model._optimizer.step()

            self.training_loss.append(loss.detach().item())
            # self.cbf_reg_loss.append(reg_loss.detach().item())

            # self.lr_schedueler.step()

            self._update_target(self._constraint_approximator, self._target_constraint_approximator)
            self._update_target(self._critic_approximator, self._target_critic_approximator)

    def _get_violations(self, dataset):
        new_data = []
        for el in dataset:
            if el[4] > 0:
                new_data.append(el[:2] + (el[4], el[3]) + el[5:])

        return new_data

    def _loss(self, state, action_new, log_prob):
        q_0 = self._critic_approximator(state, action_new, output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new, output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        return (self._alpha * log_prob - q).mean()

    def _update_alpha(self, log_prob):
        if self._use_log_alpha_loss:
            alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        else:
            alpha_loss = - (self._alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _next_q(self, next_state, absorbing):
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

        return q

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())
        self.policy._constraint_approximator = self._constraint_approximator
        self.policy._target_entropy = torch.tensor(self._target_entropy)

        self.policy._debug_cbf_bound = []
        self.policy._debug_residual_median = []
        self.policy._debug_residual_max = []
        self.policy._debug_auxiliary_action = []
        self.policy._debug_constraint_violations = []
        self.policy._debug_J_k_variance = []
        self.policy._debug_J_k_norm = []
        self.policy.device = torch.device('cpu')

        self.training_loss = []

        self._episode_cost = 0
        self._episode_end = False
        self.policy.min_a = -np.ones(3)
        self.policy.max_a = np.ones(3)

        self.policy.build_qp()

    @property
    def device(self):
        return self.policy.device

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()

    def clbf_loss(self, v, cost, u_result, r_result, L_fV, L_GV):
        unsafe_mask = cost > 0
        safe_mask = cost <= 0

        alpha_1 = 1e2
        alpha_2 = 1e2
        alpha_3 = 1

        loss_safe = torch.nan_to_num(torch.mean(F.relu(self.epsilon + v[safe_mask])))
        loss_unsafe = torch.nan_to_num(torch.mean(F.relu(self.epsilon - v[unsafe_mask])))

        loss_decent = torch.mean(r_result)

        loss = alpha_1 * loss_safe + alpha_2 * loss_unsafe + alpha_3 * loss_decent

        if torch.isnan(loss).any():
            print("NAN in loss")

        return loss
