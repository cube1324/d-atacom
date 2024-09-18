import torch.optim as optim
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import TD3, SAC
from mushroom_rl.policy import ClippedGaussianPolicy

from cremini_rl.algorithms import *
from cremini_rl.utils.networks import *


def agent_builder(alg, mdp, control_system, **kwargs):
    for key in ["n_features_actor", "n_features_critic", "n_features_constraint"]:
        if key in kwargs.keys():
            if isinstance(kwargs[key], list):
                kwargs[key] = ' '.join(map(str, kwargs[key]))

    if alg == "td3":
        return build_td3(mdp, **kwargs)

    if alg == "sac":
        return build_sac(mdp, **kwargs)

    if alg == "datacom_sac":
        return build_atacom_sac(mdp, control_system, **kwargs)

    if alg == "cbf_sac":
        return build_cbf_sac(mdp, control_system, **kwargs)

    if alg == "iqn_datacom_sac":
        return build_iqn_atacom_sac(mdp, control_system, **kwargs)

    if alg == "baseline-atacom_sac":
        return build_baseline_atacom_sac(mdp, control_system, **kwargs)

    if alg == "safelayer_td3":
        return build_safelayer_td3(mdp, **kwargs)

    if alg == "lag_sac":
        return build_lag_sac(mdp, **kwargs)

    if alg == "wc_lag_sac":
        return build_wcsac(mdp, **kwargs)


def build_baseline_atacom_sac(mdp, control_system, atacom_lam, atacom_beta, initial_replay_size, max_replay_size,
                              batch_size, n_features_actor, n_features_critic,
                              learning_rate_actor, learning_rate_critic, tau, lr_alpha, target_entropy,
                              warmup_transitions, use_viability, use_cuda,
                              **kwargs):
    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    constraint_func = mdp.constraint_func

    agent = AtacomSACBaseline(mdp.info, control_system, atacom_lam, atacom_beta, constraint_func, use_viability,
                              actor_mu_params,
                              actor_sigma_params,
                              actor_optimizer, critic_params,
                              **alg_params,
                              initial_replay_size=initial_replay_size, max_replay_size=max_replay_size,
                              batch_size=batch_size)

    return agent


def build_constraint(control_system, constraint_distribution, learning_rate_constraint,
                     n_features_constraint, use_cuda):
    if constraint_distribution == "None":
        constraint_params = dict(network=MLP,
                                 optimizer={'class': optim.Adam,
                                            'params': {'lr': learning_rate_constraint}},
                                 n_features=list(map(int, n_features_constraint.split(' '))),
                                 input_shape=(control_system.dim_q + control_system.dim_x,),
                                 output_shape=(1, 2),
                                 use_cuda=use_cuda,
                                 quiet=True,
                                 loss=F.mse_loss,
                                 activation='relu')

    elif constraint_distribution == "gaussian":
        constraint_params = dict(network=GaussianConstraintNetwork,
                                 optimizer={'class': optim.Adam,
                                            'params': {'lr': learning_rate_constraint}},
                                 n_features=list(map(int, n_features_constraint.split(' '))),
                                 input_shape=(control_system.dim_q + control_system.dim_x,),
                                 output_shape=(1, 2),
                                 use_cuda=use_cuda,
                                 n_fit_targets=5,
                                 quiet=True,
                                 activation='sigmoid')

    elif constraint_distribution == "quantile":
        constraint_params = dict(network=QuantileCriticNetwork,
                                 optimizer={'class': optim.Adam,
                                            'params': {'lr': learning_rate_constraint}},
                                 n_features=list(map(int, n_features_constraint.split(' '))),
                                 use_cuda=use_cuda,
                                 input_shape=(control_system.dim_q + control_system.dim_x,),
                                 output_shape=(1,),
                                 n_fit_targets=1,
                                 quiet=True,
                                 activation='relu')

    return constraint_params


def build_td3_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, use_cuda):
    # policy_class = OrnsteinUhlenbeckPolicy
    # policy_params = dict(sigma=np.ones(mdp.info.action_space.shape) * 0.2, theta=.15, dt=mdp.dt * mdp._n_sub_steps)

    policy_class = ClippedGaussianPolicy
    policy_params = dict(
        sigma=np.eye(mdp.info.action_space.shape[0]) * 0.25,
        low=mdp.info.action_space.low,
        high=mdp.info.action_space.high)

    # Settings
    tau = .001

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=TD3ActorNetwork,
                        n_features=list(map(int, n_features_actor.split(' '))),
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape,
                        action_scaling=(mdp.info.action_space.high - mdp.info.action_space.low) / 2,
                        use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': learning_rate_actor}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=TD3CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': learning_rate_critic}},
                         loss=F.mse_loss,
                         n_features=list(map(int, n_features_critic.split(' '))),
                         input_shape=critic_input_shape,
                         action_shape=mdp.info.action_space.shape,
                         output_shape=(1,),
                         action_scaling=(mdp.info.action_space.high - mdp.info.action_space.low) / 2,
                         use_cuda=use_cuda)

    return policy_class, policy_params, actor_params, actor_optimizer, critic_params, tau


def build_td3(mdp, initial_replay_size, max_replay_size, batch_size, n_features_actor, n_features_critic,
              learning_rate_actor, learning_rate_critic, **kwargs):
    use_cuda = False

    policy_class, policy_params, actor_params, actor_optimizer, critic_params, tau = build_td3_params(
        mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, use_cuda)

    agent = TD3(mdp.info, policy_class, policy_params,
                actor_params, actor_optimizer, critic_params, batch_size,
                initial_replay_size, max_replay_size, tau)

    return agent


def build_safelayer_td3(mdp, initial_replay_size, max_replay_size, batch_size, n_features_actor, n_features_critic,
                        learning_rate_actor, learning_rate_critic, learning_rate_constraint, n_features_constraint,
                        delta, warmup_transitions, **kwargs):
    use_cuda = False

    constraint_params = dict(network=MLP,
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': learning_rate_constraint}},
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape,
                             n_features=list(map(int, n_features_constraint.split(' '))),
                             loss=SafeLayerTD3.safelayer_loss,
                             use_cuda=False,
                             quiet=True,
                             n_fit_targets=3)

    policy_class, policy_params, actor_params, actor_optimizer, critic_params, tau = build_td3_params(
        mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, use_cuda)

    policy_class = ClippedGaussianPolicy

    agent = SafeLayerTD3(mdp.info, policy_class, policy_params, actor_params, actor_optimizer, critic_params,
                         batch_size,
                         initial_replay_size, max_replay_size, tau, delta, constraint_params, warmup_transitions)

    return agent


def build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, use_cuda,
                     tau, lr_alpha, target_entropy, warmup_transitions):
    actor_mu_params = dict(network=SACActorNetwork,
                           input_shape=mdp.info.observation_space.shape,
                           output_shape=mdp.info.action_space.shape,
                           n_features=list(map(int, n_features_actor.split(' '))),
                           use_cuda=use_cuda)
    actor_sigma_params = dict(network=SACActorNetwork,
                              input_shape=mdp.info.observation_space.shape,
                              output_shape=mdp.info.action_space.shape,
                              n_features=list(map(int, n_features_actor.split(' '))),
                              use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': learning_rate_actor}}

    critic_params = dict(network=SACCriticNetwork,
                         input_shape=(mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0],),
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': learning_rate_critic}},
                         loss=F.mse_loss,
                         n_features=list(map(int, n_features_critic.split(' '))),
                         output_shape=(1,),
                         action_shape=mdp.info.action_space.shape,
                         action_scaling=(mdp.info.action_space.high - mdp.info.action_space.low) / 2,
                         use_cuda=use_cuda
                         )

    alg_params = dict(warmup_transitions=warmup_transitions,
                      tau=tau,
                      lr_alpha=lr_alpha,
                      target_entropy=target_entropy)

    return actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params


def build_sac(mdp, initial_replay_size, max_replay_size, batch_size, n_features_actor, n_features_critic,
              learning_rate_actor, learning_rate_critic, use_cuda, tau, lr_alpha, target_entropy, warmup_transitions,
              **kwargs):
    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    print(alg_params, use_cuda)

    agent = SAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, **alg_params,
                initial_replay_size=initial_replay_size, max_replay_size=max_replay_size,
                batch_size=batch_size)

    return agent


def build_atacom_sac(mdp, control_system, initial_replay_size, max_replay_size, batch_size, n_features_actor,
                     n_features_critic, n_features_constraint, learning_rate_actor, learning_rate_critic,
                     accepted_risk, learning_rate_constraint,
                     atacom_lam, atacom_beta, use_cuda, tau, lr_alpha, target_entropy,
                     warmup_transitions, cost_budget, lr_delta, init_delta, delta_warmup_transitions, **kwargs):
    constraint_params = build_constraint(control_system, "gaussian",
                                         learning_rate_constraint, n_features_constraint, use_cuda)

    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    if hasattr(mdp, "analytical_constraint"):
        alg_params["analytical_constraint"] = mdp.analytical_constraint

    agent = GaussianAtacomSAC(mdp_info=mdp.info, control_system=control_system, accepted_risk=accepted_risk,
                              actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                              actor_optimizer=actor_optimizer, critic_params=critic_params, batch_size=batch_size,
                              initial_replay_size=initial_replay_size, max_replay_size=max_replay_size,
                              cost_budget=cost_budget, constraint_params=constraint_params, atacom_lam=atacom_lam,
                              atacom_beta=atacom_beta, lr_delta=lr_delta, init_delta=init_delta,
                              delta_warmup_transitions=delta_warmup_transitions,
                              **alg_params)

    return agent


def build_cbf_sac(mdp, control_system, initial_replay_size, max_replay_size, batch_size, n_features_actor,
                  n_features_critic, n_features_constraint, learning_rate_actor, learning_rate_critic,
                  learning_rate_constraint,
                  use_cuda, tau, lr_alpha, target_entropy,
                  warmup_transitions, **kwargs):
    constraint_params = dict(network=MLP,
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': learning_rate_constraint}},
                             n_features=list(map(int, n_features_constraint.split(' '))),
                             input_shape=(control_system.dim_q + control_system.dim_x,),
                             output_shape=(1,),
                             use_cuda=use_cuda,
                             quiet=True,
                             loss=F.mse_loss,
                             activation='relu')

    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    if hasattr(mdp, "analytical_constraint"):
        alg_params["analytical_constraint"] = mdp.analytical_constraint

    agent = CBFSAC(mdp_info=mdp.info, control_system=control_system,
                   actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                   actor_optimizer=actor_optimizer, critic_params=critic_params, batch_size=batch_size,
                   initial_replay_size=initial_replay_size, max_replay_size=max_replay_size,
                   constraint_params=constraint_params,
                   **alg_params)

    return agent


def build_iqn_atacom_sac(mdp, control_system, initial_replay_size, max_replay_size, batch_size, n_features_actor,
                         n_features_critic, n_features_constraint, learning_rate_actor, learning_rate_critic,
                         learning_rate_constraint, accepted_risk,
                         atacom_lam, atacom_beta, use_cuda, tau, lr_alpha, target_entropy, warmup_transitions,
                         quantile_embedding_dim, num_quantile_samples,
                         num_next_quantile_samples,
                         cost_budget, lr_delta, init_delta, delta_warmup_transitions, **kwargs):
    constraint_params = build_constraint(control_system, 'quantile',
                                         learning_rate_constraint, n_features_constraint, use_cuda)

    constraint_params['embedding_size'] = quantile_embedding_dim

    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    agent = IQNAtacomSAC(mdp_info=mdp.info, control_system=control_system, accepted_risk=accepted_risk,
                         actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                         actor_optimizer=actor_optimizer,
                         critic_params=critic_params, batch_size=batch_size, initial_replay_size=initial_replay_size,
                         max_replay_size=max_replay_size, cost_budget=cost_budget, constraint_params=constraint_params,
                         atacom_lam=atacom_lam, atacom_beta=atacom_beta,
                         lr_delta=lr_delta, init_delta=init_delta, delta_warmup_transitions=delta_warmup_transitions,
                         num_quantile_samples=num_quantile_samples, num_next_quantile_samples=num_next_quantile_samples,
                         **alg_params)
    return agent


def build_lag_sac(mdp, initial_replay_size, max_replay_size, batch_size, n_features_actor, n_features_critic,
                  learning_rate_actor, learning_rate_critic, learning_rate_constraint, tau, lr_alpha, target_entropy,
                  warmup_transitions, use_cuda, lr_beta, cost_limit, damp_scale, **kwargs):
    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    constraint_params = dict(network=SACCriticNetwork,
                             input_shape=(mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0],),
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': learning_rate_constraint}},
                             loss=F.mse_loss,
                             n_features=list(map(int, n_features_critic.split(' '))),
                             output_shape=(1,),
                             use_cuda=use_cuda)

    agent = LagSAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, **alg_params,
                   constraint_params=constraint_params, initial_replay_size=initial_replay_size,
                   max_replay_size=max_replay_size, batch_size=batch_size,
                   lr_beta=lr_beta, cost_limit=cost_limit, damp_scale=damp_scale)

    return agent


def build_wcsac(mdp, initial_replay_size, max_replay_size, batch_size, n_features_actor, n_features_critic,
                learning_rate_actor, learning_rate_critic, learning_rate_constraint, accepted_risk, tau, lr_alpha,
                target_entropy, warmup_transitions, lr_beta, cost_limit, damp_scale, constraint_type, **kwargs):
    use_cuda = False

    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    if constraint_type == "gaussian":
        net = GaussianConstraintQNetwork
    elif constraint_type == "quantile":
        net = ImplicitQuantileConstraint

    constraint_params = dict(network=net,
                             input_shape=(mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0],),
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': learning_rate_constraint}},
                             n_features=list(map(int, n_features_critic.split(' '))),
                             output_shape=(1,),
                             use_cuda=use_cuda,
                             embedding_dim=n_features_critic,
                             num_cosines=n_features_critic)

    agent = WCSAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, **alg_params,
                  constraint_params=constraint_params, initial_replay_size=initial_replay_size,
                  max_replay_size=max_replay_size, batch_size=batch_size,
                  lr_beta=lr_beta, cost_limit=cost_limit, accepted_risk=accepted_risk, damp_scale=damp_scale,
                  constraint_type=constraint_type)

    return agent
