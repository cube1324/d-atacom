import omnisafe
from cremini_rl.envs.omnisafe_wrapper import OmnisafeWrapper

import numpy as np
import os
import csv   

def eval(path):
    evaluator = omnisafe.Evaluator(render_mode='human')

    try:
        evaluator.load_saved(
            save_dir=path, model_name="epoch-500.pt"
        )
        
    except:
        evaluator.load_saved(
            save_dir=path, model_name="epoch-200.pt"
        )
        

    R, J, cost, episode_lengths = evaluator.evaluate(num_episodes=20)

    mean_R = np.mean(R)
    mean_J = np.mean(J)
    cost = np.array(cost)
    episode_lengths = np.array(episode_lengths)
    mean_cost = np.mean(cost / episode_lengths)
    sum_cost = np.mean(cost)

    return mean_R, mean_J, episode_lengths, mean_cost, sum_cost


def eval_exp(base_path):
    path = os.path.join(base_path, "0", "Benchmark")
    
    path = os.path.join(path, get_folder(path))
    path = os.path.join(path, get_folder(path))
    seed_R = []
    seed_J = []
    seed_episode_length = []
    seed_cost = []
    seed_sum_cost = []

    for seed in os.listdir(path):
        if os.path.isdir(os.path.join(path, seed)):
            seed_path = os.path.join(path, seed)
            R, J, episode_length, cost, sum_cost = eval(seed_path)
            seed_R.append(R)
            seed_J.append(J)
            seed_episode_length.append(episode_length)
            seed_cost.append(cost)
            seed_sum_cost.append(sum_cost)
    
    return np.mean(seed_R), np.mean(seed_J), np.mean(seed_episode_length), np.mean(seed_cost), np.mean(seed_sum_cost)


def eval_configs(base_path):
    for alg in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, alg)) and alg.startswith("algo"): # and alg.endswith("PPOLag"):
            alg_path = os.path.join(base_path, alg)
            for lr in os.listdir(alg_path):
                if os.path.isdir(os.path.join(alg_path, lr)): # and lr.endswith("0.0001"):
                    lr_path = os.path.join(alg_path, lr, "cost_limit___0")
                    print(lr_path)
                    mean_R, mean_J, mean_episode_length, mean_cost, mean_sum_cost = eval_exp(lr_path)

                    with open(r'tiago_navigation.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([alg[7:], lr[5:], mean_R, mean_J, mean_episode_length, mean_cost, mean_sum_cost])

                    print(f"Algorithm: {alg}, LR: {lr}")
                    print(f"Mean R: {mean_R}")
                    print(f"Mean J: {mean_J}")
                    print(f"Mean Episode Length: {mean_episode_length}")
                    print(f"Mean Cost: {mean_cost}")
                    print(f"Mean Sum Cost: {mean_sum_cost}")
                    print("\n")
                    




def get_folder(path):
    for el in os.listdir(path):
        if os.path.isdir(os.path.join(path, el)):
            return el

if __name__ == "__main__":
    data = eval_configs("../logs/omnisafe-tiago_navigation_2024-05-10_23-11-29")
    print(data)
