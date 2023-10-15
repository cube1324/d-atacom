import numpy as np
import torch

from cremini_rl.envs import *

from cremini_rl.algorithms import *

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC
from mushroom_rl.core import Core

from cremini_rl.utils.safe_core import SafeCore

safe_agent = True
# path_to_agent = "../logs/cartpole_2024-05-01_14-49-01/0/agent-102.msh"
path_to_agent = "agent_cbf.msh"
if safe_agent:
    # agent = WCSAC.load(path_to_agent)
    # agent = GaussianAtacomSAC.load(path_to_agent)
    agent = CBFSAC.load(path_to_agent)
else:
    agent = SAC.load(path_to_agent)

# np.random.seed(18)
# torch.manual_seed(18)

# mdp = SafeCartPoleEnv(horizon=agent.mdp_info.horizon, gamma=agent.mdp_info.gamma, return_cost=safe_agent)
# mdp = TiagoNavigationEnv(horizon=agent.mdp_info.horizon, gamma=agent.mdp_info.gamma, return_cost=safe_agent, render=True)
# mdp = DockingEnv(horizon=agent.mdp_info.horizon, gamma=agent.mdp_info.gamma, return_cost=True)
mdp = PlanarAirhockeyEnv(return_cost=safe_agent)
# mdp = GoalNavigationEnv(horizon=agent.mdp_info.horizon, gamma=agent.mdp_info.gamma, return_cost=safe_agent, static=True, render=True)

if safe_agent:
    core = SafeCore(agent, mdp)
else:
    core = Core(agent, mdp)

core.evaluate(n_episodes=20, render=True)

