from experiment_launcher import Launcher, is_local

from itertools import product

import hiyapyco
import os


def main():
    # Choices: cartpole, tiago_navigation, planar_air_hockey, dense_ball2d, goal_navigation,
    env = "planar_air_hockey"
    # Choices: sac, td3, datacom_sac, iqn_datacom_sac, safelayer_td3, lag_sac, wc_lag_sac, cbf_sac, baseline-atacom_sac
    alg = "iqn_datacom_sac"

    # Load configs based on the algorithm and environment, there are defaults for each algorithm. They are merged
    # with the environment specific config if it exists.
    configs = [os.path.join("configs", "defaults", f"{part}.yaml") for part in alg.split("_")]

    if os.path.exists(os.path.join("configs", f"{alg}_{env}.yaml")):
        configs.append(os.path.join("configs", f"{alg}_{env}.yaml"))

    config = hiyapyco.load(os.path.join("configs", "defaults", "defaults.yaml"), *configs)

    # Check if on a slurm cluster or local machine
    local = is_local()
    if local:
        debug = False
        use_wandb = False
        n_seeds = 1
        n_exp_in_parallel = 1
    else:
        debug = False
        use_wandb = True
        n_seeds = 10
        n_exp_in_parallel = 1

    launcher = Launcher(env, f"experiment", n_seeds=n_seeds, memory_per_core=1500, n_cores=1,
                        conda_env="safe_rl", hours=30, n_exps_in_parallel=n_exp_in_parallel)

    keys = []
    values = []
    for key, value in config.items():
        if type(value) is list:
            keys.append(key)
            values.append(value)

    if len(values) == 0:
        keys.append(next(iter(config.keys())))
        values.append([config[keys[0]]])

    for setting in product(*values):
        for i, (key, value) in enumerate(zip(keys, values)):
            update_param(config, key, setting[i], value)

        wandb_options = dict(
            wandb_enabled=use_wandb,
            wandb_entity='jonascodes',
            wandb_project=env,
            wandb_group=f"{alg}",
        )

        assert not " " in wandb_options["wandb_group"], "NO SPACE IN GROUP NAME"

        launcher.add_experiment(env_name=env, alg=alg, debug=debug, n_exp_in_parallel=n_exp_in_parallel, **config,
                                **wandb_options)

    launcher.run(local, False)


def update_param(config, key, value, value_list):
    if len(value_list) > 1:
        config.pop(key, None)
        config[f"{key}__"] = value
    else:
        config[key] = value


if __name__ == "__main__":
    main()
