from experiment_launcher import Launcher, is_local
import cremini_rl.envs.omnisafe_wrapper


def main():
    env = "planar_air_hockey"
    # algo = "PCPO"
    n_seeds = 10
    n_cores = 10

    local = is_local()

    launcher = Launcher(f"omnisafe-{env}", f"omnisafe_experiment", n_seeds=1, memory_per_core=1500, n_cores=n_cores,
                        conda_env="safe_rl", hours=23, n_exps_in_parallel=1, partition="stud")

    lr_rates = [5e-4]
    cost_limits = [1]
    # algs = ["CPO", "PCPO", 'PPOLag', 'TRPOLag', 'RCPO', 'OnCRPO', 'PDO']
    algs = ["SafeLOOP"]
    # algs = ["CAPPETS"] #, "SafeLOOP"]

    for lr in lr_rates:
        for cost_limit in cost_limits:
            for algo in algs:
                launcher.add_experiment(env_id=env, n_seeds=n_seeds, n_cores=n_cores, algo__=algo, lr__=lr,
                                        cost_limit__=cost_limit)

    launcher.run(local, False)


if __name__ == "__main__":
    main()
