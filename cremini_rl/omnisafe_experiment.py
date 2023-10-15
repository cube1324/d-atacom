import omnisafe
from cremini_rl.envs.omnisafe_wrapper import OmnisafeWrapper
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train

from experiment_launcher import run_experiment, single_experiment


@single_experiment
def experiment(seed: int = 0,
               results_dir: str = 'logs',
               n_seeds: int = 10,
               n_cores: int = 10,
               env_id: str = 'cartpole',
               algo: str = 'CPO',
               lr: float = 5e-4,
               cost_limit: float = 10,
               **kwargs):
    eg = ExperimentGrid(exp_name='Benchmark')
    eg.add('env_id', [env_id])

    eg.add('algo', [algo])

    if algo == "CAPPETS":
        eg.add('algo_cfgs:action_repeat', [1])
        eg.add('logger_cfgs:use_wandb', [True])
        eg.add('logger_cfgs:wandb_project', [f"omnisafe-{env_id}"])
        eg.add('train_cfgs:vector_env_nums', [1])
        eg.add('train_cfgs:torch_threads', [2])
        eg.add('lagrange_cfgs:cost_limit', [cost_limit])
        eg.add('evaluation_cfgs:use_eval', [False])
        eg.add('dynamics_cfgs:num_ensemble', [2])
        eg.add('dynamics_cfgs:elite_size', [2])
        eg.add('planner_cfgs:num_particles', [4])
        eg.add('planner_cfgs:num_samples', [64])
        eg.add('planner_cfgs:plan_horizon', [3])
        eg.add('planner_cfgs:num_elites', [8])
        eg.add('planner_cfgs:num_iterations', [3])


    else:
        # eg.add('algo_cfgs:batch_size', [32])
        # eg.add('algo_cfgs:steps_per_epoch', [2000])
        # eg.add('algo_cfgs:update_iters', [10])
        # eg.add('algo_cfgs:entropy_coef', [1e-2])
        eg.add('algo_cfgs:auto_alpha', [True])
        # eg.add('algo_cfgs:action_repeat', [1])
        eg.add('logger_cfgs:use_wandb', [True])
        eg.add('logger_cfgs:wandb_project', [f"omnisafe-{env_id}"])
        eg.add('train_cfgs:vector_env_nums', [1])
        eg.add('train_cfgs:torch_threads', [1])
        eg.add('model_cfgs:actor:lr', [lr])
        eg.add('model_cfgs:critic:lr', [lr])
        eg.add('model_cfgs:actor:hidden_sizes', [[128, 128]])
        eg.add('model_cfgs:critic:hidden_sizes', [[128, 128]])
        eg.add('evaluation_cfgs:use_eval', [False])

        if algo in ["CPO", "PCPO", 'OnCRPO', "SafeLOOP"]:
            eg.add('algo_cfgs:cost_limit', [cost_limit])
        else:
            eg.add('lagrange_cfgs:cost_limit', [cost_limit])
        # eg.add('algo_cfgs:steps_per_epoch', [1000])
        eg.add('train_cfgs:total_steps', [1000000])
        eg.add('evaluation_cfgs:eval_cycle', [20000])
    # eg.add('planner_cfgs:init_var', [0.1])
    eg.add('seed', list(range(n_seeds)))
    # total experiment num must can be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=n_cores, parent_dir=results_dir, is_test=False)  # , gpu_id=gpu_id)

    # eg.evaluate(n_episodes=50)


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)
