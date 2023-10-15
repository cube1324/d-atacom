from safety_gymnasium.utils.registration import register
from cremini_rl.envs.custom_safety_gymnasium.static_goal_navigation import StaticGoalEnv
from cremini_rl.envs.custom_safety_gymnasium.dense_goal_navigation import DenseGoalLevel1

register(id='StaticGoalNavigation-v0', entry_point='cremini_rl.envs.custom_safety_gymnasium.better_builder:BetterBuilder',
         kwargs={"task_class": StaticGoalEnv, 'config':{'agent_name': 'Point'}})

register(id='DenseGoalNavigation1-v0', entry_point='cremini_rl.envs.custom_safety_gymnasium.better_builder:BetterBuilder',
         kwargs={"task_class": DenseGoalLevel1, 'config':{'agent_name': 'Point'}})