import torch

from mushroom_rl.core import Logger, Core
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length, parse_dataset, get_init_states

from cremini_rl.envs import *
from cremini_rl.envs.tiago_navigation_env import TiagoNavigationEnv
from cremini_rl.envs.cartpole_goal_env import SafeCartPoleEnv

from cremini_rl.utils.safe_core import SafeCore
from cremini_rl.dynamics import *

from cremini_rl.utils.agent_builder import agent_builder

from experiment_launcher.decorators import single_experiment

import os
import wandb
import argparse

import numpy as np
from copy import deepcopy

from joblib import Parallel, delayed


@single_experiment
def experiment(results_dir: str,
               seed: int,
               env_name: str,
               alg: str,
               n_epochs: int,
               n_steps: int,
               n_episodes_test: int,
               initial_replay_size: int,
               quiet: bool,
               render: bool,

               debug: bool,
               **kwargs
               ):
    seed += 0

    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    logger = Logger(log_name=".", results_dir=results_dir)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg)

    return_cost = "atacom" in alg or "safelayer" in alg or "lag" in alg or alg == "wcsac" or alg == "cbf_sac"

    mdp, control_system = build_mdp(env_name, return_cost)

    gamma = mdp.info.gamma

    agent = agent_builder(alg, mdp, control_system, initial_replay_size=initial_replay_size, **kwargs)

    if return_cost:
        core = SafeCore(agent, mdp)
    else:
        core = Core(agent, mdp)

    # LEarn dynamics
    # core.learn(n_steps=52000, n_steps_per_fit=1, quiet=quiet, render=render)

    # RUN
    data = evaluate(core, n_episodes_test, gamma, quiet, render=render)

    best_J = data["J"]

    log_data(data, 0, logger)

    for n in range(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=quiet, render=render)

        data = evaluate(core, n_episodes_test, gamma, quiet, render=render)
        if data["J"] > best_J:
            best_J = data["J"]
            logger.log_agent(agent, "J")

        if n % 25 == 0:
            logger.log_agent(agent, n)

        log_data(data, n + 1, logger)

        logger.log_agent(agent)

    logger.log_agent(agent)

    data = evaluate(core, n_episodes_test, gamma, quiet, render=render)
    log_data(data, n + 2, logger)

    wandb.save(os.path.join(logger.path, "*.msh"), logger.path)
    wandb.save(os.path.join(logger.path, "*.npy"), logger.path)


def evaluate(core, n_episodes_test, gamma, quiet, render):
    dataset, info = core.evaluate(n_episodes=n_episodes_test, render=render, get_env_info=True, quiet=quiet)

    episode_length = compute_episodes_length(dataset)
    init_states = get_init_states(dataset)
    states = np.array([d[0] for d in dataset])

    J = np.mean(compute_J(dataset, gamma))
    R = np.mean(compute_J(dataset))
    cost = np.array(info["cost"])

    mean_cost = []
    sum_cost = []
    max_violation = []

    epi_idx = 0
    for epi in episode_length:
        epi_cost = cost[epi_idx:epi_idx + epi]
        epi_cost = np.maximum(epi_cost, 0)
        max_violation.append(np.max(epi_cost))
        sum_cost.append(np.sum(epi_cost))
        mean_cost.append(np.mean(epi_cost))
        epi_idx += epi

    mean_cost = np.mean(mean_cost)
    sum_cost = np.mean(sum_cost)
    max_violation = np.mean(max_violation)

    data_dict = dict(J=J, R=R, mean_cost=mean_cost, sum_cost=sum_cost, max_violation=max_violation,
                     episode_length=np.mean(episode_length))

    if hasattr(core.agent, "_critic_approximator"):
        data_dict["V"] = compute_V(core.agent, init_states, core.agent._critic_approximator)

    if hasattr(core.agent, "_alpha"):
        data_dict["alpha"] = core.agent._alpha.detach().cpu().numpy().item()

    if hasattr(core.agent.policy, "entropy"):
        E = core.agent.policy.entropy(states)
        data_dict["E"] = E

    if "success" in info.keys():
        data_dict["success_rate"] = np.sum(info["success"]) / len(episode_length)

    if "delta_v" in info.keys():
        data_dict["delta_v"] = np.mean(info["delta_v"])

    if hasattr(core.agent, "training_loss") and len(core.agent.training_loss) > 0:
        training_loss = np.mean(core.agent.training_loss)
        data_dict["cbf_loss"] = training_loss
        core.agent.training_loss = []

    if hasattr(core.agent, "cbf_reg_loss") and len(core.agent.cbf_reg_loss) > 0:
        data_dict["cbf_reg_loss"] = np.mean(core.agent.cbf_reg_loss)
        core.agent.cbf_reg_loss = []

    if hasattr(core.agent.policy, "_debug_constraint_violations") and len(
            core.agent.policy._debug_constraint_violations) > 0:
        violation_records = core.agent.policy._debug_constraint_violations
        voilation_mean = np.mean(violation_records, axis=0)
        voilation_max = np.max(violation_records, axis=0)
        if core.agent.policy._learn_constraint:
            data_dict["mean_learned_constraint"] = voilation_mean[0]
            data_dict["max_learned_constraint"] = voilation_max[0]
            if core.agent.policy._learn_cbf:
                data_dict["mean_learned_cbf"] = voilation_mean[1]
                data_dict["max_learned_cbf"] = voilation_max[1]
        elif core.agent.policy._learn_cbf:
            data_dict["mean_learned_cbf"] = voilation_mean[0]
            data_dict["max_learned_cbf"] = voilation_max[0]
            data_dict["cbf_bound"] = np.mean(core.agent.policy._debug_cbf_bound)

        data_dict["J_k_variance"] = np.mean(core.agent.policy._debug_J_k_variance)
        data_dict["J_k_norm"] = np.mean(core.agent.policy._debug_J_k_norm)

        core.agent.policy._debug_J_k_variance = []
        core.agent.policy._debug_J_k_norm = []

        core.agent.policy._debug_constraint_violations = []
        core.agent.policy._debug_cbf_bound = []

    if hasattr(core.agent.policy, "_debug_residual_median") and len(core.agent.policy._debug_residual_median) > 0:
        residual = np.mean(core.agent.policy._debug_residual_median, axis=0)
        data_dict["residual_median"] = residual
        core.agent.policy._debug_residual_median = []

    if hasattr(core.agent.policy, "_debug_residual_max") and len(core.agent.policy._debug_residual_max) > 0:
        residual = np.mean(core.agent.policy._debug_residual_max, axis=0)
        data_dict["residual_max"] = residual
        core.agent.policy._debug_residual_max = []

    if hasattr(core.agent.policy, "_debug_residual") and len(core.agent.policy._debug_residual) > 0:
        residual_mean = np.mean(core.agent.policy._debug_residual, axis=0)
        residual_max = np.max(core.agent.policy._debug_residual, axis=0)
        data_dict["residual_mean"] = residual_mean
        data_dict["residual_max"] = residual_max
        core.agent.policy._debug_residual = []

    if hasattr(core.agent.policy, "accepted_risk"):
        data_dict["accepted_risk"] = core.agent.policy.accepted_risk().detach().numpy()
        constraint_state, _ = core.agent.to_constraint_state(states, states)
        tau = torch.ones(constraint_state.shape[0], 2) * torch.tensor([0.05, 0.95])
        var = core.agent._constraint_approximator.predict(constraint_state, tau)
        data_dict["var_span"] = np.mean(var[:, 1] - var[:, 0])

    if hasattr(core.agent.policy, "_debug_importance_weights") and len(core.agent.policy._debug_importance_weights) > 0:
        data_dict["importance_weight"] = wandb.Histogram(
            np.array(core.agent.policy._debug_importance_weights).flatten())
        core.agent.policy._debug_importance_weights = []

    if hasattr(core.agent, "_constraint_approximator"):
        if hasattr(core.agent, "_num_quantile_samples"):
            constraint_init_states, _ = core.agent.to_constraint_state(init_states, init_states)
            tau = torch.ones(constraint_init_states.shape[0], 1) - core.agent.policy.accepted_risk()
            cbf = core.agent._constraint_approximator.predict(constraint_init_states, tau)
            data_dict["cbf"] = np.mean(cbf)

        elif hasattr(core.agent, "_delta_value"):
            constraint_init_states, _ = core.agent.to_constraint_state(init_states, init_states)
            mean, std = core.agent._constraint_approximator.predict(constraint_init_states)
            data_dict["cbf"] = np.mean(mean)
            data_dict["cbf_std"] = np.mean(std)

        elif hasattr(core.agent, "to_constraint_state"):
            constraint_init_states, _ = core.agent.to_constraint_state(init_states, init_states)
            data_dict["cbf"] = np.mean(core.agent._constraint_approximator.predict(constraint_init_states))

    if hasattr(core.agent.policy, "_debug_auxiliary_action") and len(core.agent.policy._debug_auxiliary_action) > 0:
        value = np.mean(core.agent.policy._debug_auxiliary_action, axis=0)
        data_dict["drift_comp_action_drift"] = value[0]
        data_dict["contraction_action"] = value[1]
        core.agent.policy._debug_auxiliary_action = []

    if hasattr(core.agent.policy, "_max_seen_constraint"):
        data_dict["max_seen_const"] = core.agent.policy._max_seen_constraint

    if hasattr(core.agent.policy, "_tolerance_cbf"):
        data_dict["tolerance_cbf"] = core.agent.policy._delta.detach().numpy()

    if hasattr(core.agent, "delta"):
        data_dict["delta"] = core.agent.delta().detach().numpy()

    return data_dict


def compute_V(agent, states, approx):
    Q = list()
    for state in states:
        s = np.array([state for i in range(10)])
        a = np.array([agent.policy.draw_action(state) for i in range(10)])
        Q.append(approx(s, a).mean())
    return np.array(Q).mean()


def get_init_states(dataset):
    pick = True
    x_0 = list()
    for d in dataset:
        if pick:
            x_0.append(d[0])
        pick = d[-1]
    return np.array(x_0)


def log_data(data, episode, logger):
    logger.epoch_info(episode, **data)
    logger.log_numpy(**data)
    wandb.log(data, step=episode)


def build_mdp(env_name, return_cost):
    if env_name == "ball2d":
        mdp = BallND(n=2, return_cost=return_cost)

        control_system = VelocityControlSystem(2, list(range(2)), 1)

    if env_name == "dense_ball2d":
        mdp = BallND(n=2, return_cost=return_cost, dense_const=True)

        control_system = VelocityControlSystem(2, list(range(2)), 1)

    elif env_name == "tiago_navigation":
        mdp = TiagoNavigationEnv(return_cost=return_cost)

        control_system = TiagoNavigationDynamics()

    elif env_name == "cartpole":
        mdp = SafeCartPoleEnv(return_cost=return_cost)

        dynamics_info = {'mc': mdp.base_env._model.body('cart').mass[0],
                         'mp': mdp.base_env._model.body('pole_1').mass[0],
                         'l': mdp.base_env._model.geom('pole_1').size[1],
                         'g': -mdp.base_env._model.opt.gravity[2],
                         'u_limit': mdp.base_env._model.actuator_gear[0, 0] * 0.95,
                         'Jp': mdp.base_env._model.body_inertia[2, 1]}

        control_system = CartPoleControlSystem(**dynamics_info)

    elif env_name == "air_hockey":
        mdp = AirHockeyEnv(return_cost=return_cost)

        q_idx = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        control_system = AccelerationControlSystem(7, q_idx, 1)

    elif env_name == "planar_air_hockey":
        mdp = PlanarAirhockeyEnv(return_cost=return_cost, dynamic_noise=0)

        q_idx = [6, 7, 8, 9, 10, 11]
        control_system = AccelerationControlSystem(3, q_idx, 1)

    elif env_name == "goal_navigation":
        mdp = GoalNavigationEnv(return_cost=return_cost)

        control_system = GoalNavigationControlSystem(vases=False)

    elif env_name == "static_goal_navigation":
        mdp = GoalNavigationEnv(static=True, return_cost=return_cost)

        control_system = GoalNavigationControlSystem(vases=False)


    elif env_name == "moving_obs_2d":
        mdp = MovingObsEnv(return_cost=return_cost, random_obs=True)

        q_idx = [0, 1]
        x_idx = 2 + np.arange(2 * mdp.n_obs)
        x_dot_idx = x_idx[-1] + 1 + np.arange(2 * mdp.n_obs)
        control_system = MovingObsDynamics(2, q_idx, x_idx, x_dot_idx, mdp.vel_limits)

    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return mdp, control_system


def parse_args():
    parser = argparse.ArgumentParser()

    arg_exp = parser.add_argument_group('Experiment')

    arg_exp.add_argument("--env_name", type=str)
    arg_exp.add_argument("--alg", choices=["sac", "td3", "datacom_sac", 'iqn_datacom_sac',
                                           "safelayer_td3", "lag_sac", "wc_lag_sac",
                                           'cbf_sac', "baseline-atacom_sac"])

    arg_exp.add_argument("--n_epochs", type=int)
    arg_exp.add_argument("--n_steps", type=int)
    arg_exp.add_argument("--n_episodes_test", type=int)
    arg_exp.add_argument("--quiet", type=lambda x: x.lower() == "true")
    arg_exp.add_argument("--render", type=lambda x: x.lower() == "true")
    arg_exp.add_argument("--use_cuda", type=lambda x: x.lower() == "true")
    arg_exp.add_argument("--use_viability", type=lambda x: x.lower() == "true")

    arg_exp.add_argument("--learning_rate_actor", type=float)
    arg_exp.add_argument("--learning_rate_critic", type=float)
    arg_exp.add_argument("--learning_rate_constraint", type=float)

    arg_exp.add_argument("--n_features_actor", type=int, nargs='+')
    arg_exp.add_argument("--n_features_critic", type=int, nargs='+')
    arg_exp.add_argument("--n_features_constraint", type=int, nargs='+')

    arg_exp.add_argument("--initial_replay_size", type=int)
    arg_exp.add_argument("--max_replay_size", type=int)
    arg_exp.add_argument("--batch_size", type=int)

    arg_exp.add_argument("--accepted_risk", type=float)
    arg_exp.add_argument("--learning_strategy", type=str)
    arg_exp.add_argument("--cbf_gamma", type=float)
    arg_exp.add_argument("--margin_type", type=str)

    arg_exp.add_argument("--constraint_init_size", type=int)
    arg_exp.add_argument("--constraint_max_size", type=int)
    arg_exp.add_argument("--constraint_batch_size", type=int)

    arg_exp.add_argument("--tau", type=float)

    # SAC
    arg_exp.add_argument("--lr_alpha", type=float)
    arg_exp.add_argument("--warmup_transitions", type=float)
    arg_exp.add_argument("--target_entropy", type=float)

    # Atacom
    arg_exp.add_argument("--atacom_lam", type=float)
    arg_exp.add_argument("--atacom_beta", type=float)
    arg_exp.add_argument("--cost_budget", type=float)
    arg_exp.add_argument("--lr_delta", type=float)
    arg_exp.add_argument("--init_delta", type=float)
    arg_exp.add_argument("--delta_warmup_transitions", type=int)

    # IQN
    arg_exp.add_argument("--quantile_embedding_dim", type=int)
    arg_exp.add_argument("--num_quantile_samples", type=int)
    arg_exp.add_argument("--num_next_quantile_samples", type=int)

    # SafeLayer
    arg_exp.add_argument("--delta", type=float)

    # Lagranien SAC
    arg_exp.add_argument("--lr_beta", type=float)
    arg_exp.add_argument("--cost_limit", type=float)
    arg_exp.add_argument("--damp_scale", type=float)

    # WCSAC
    arg_exp.add_argument("--constraint_type", type=str)

    arg_exp.add_argument("--debug", type=lambda x: x.lower() == "true")
    arg_exp.add_argument("--seed", type=int, default=0)
    arg_exp.add_argument("--n_exp_in_parallel", type=int)
    arg_exp.add_argument("--results_dir", type=str)

    arg_exp.add_argument("--wandb_enabled", type=lambda x: x.lower() == "true")
    arg_exp.add_argument("--wandb_entity", type=str)
    arg_exp.add_argument("--wandb_project", type=str)
    arg_exp.add_argument("--wandb_group", type=str)

    args, unknown = parser.parse_known_args()

    return vars(args)


if __name__ == '__main__':
    args = parse_args()

    global_seed = args["seed"]
    del args["seed"]

    Parallel(n_jobs=5)(delayed(experiment)(**deepcopy(args), seed=i + global_seed * args["n_exp_in_parallel"]) for i in
                       range(args["n_exp_in_parallel"]))
