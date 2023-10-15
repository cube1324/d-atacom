from safety_gymnasium.tasks.goal.goal_level0 import GoalLevel0
from safety_gymnasium.assets.geoms import Hazards, Goal


'''
config1 = {
        'placements_extents': [-1.5, -1.5, 1.5, 1.5],
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'goal_size': 0.3,
        'goal_keepout': 0.305,
        'goal_locations': [(1.1, 1.1)],
        'observe_goal_lidar': True,
        'observe_hazards': True,
        'constrain_hazards': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'hazards_num': 1,
        'hazards_size': 0.7,
        'hazards_keepout': 0.705,
        'hazards_locations': [(0, 0)]
        }


register(id='StaticEnv-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config1})

config2 = {
        'placements_extents': [-1.5, -1.5, 1.5, 1.5],
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'goal_size': 0.3,
        'goal_keepout': 0.305,
        'observe_goal_lidar': True,
        'observe_hazards': True,
        'constrain_hazards': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'hazards_num': 3,
        'hazards_size': 0.3,
        'hazards_keepout': 0.305
        }

register(id='DynamicEnv-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config2})
'''

class StaticGoalEnv(GoalLevel0):
    """An agent must navigate to a goal while avoiding hazards.

    One vase is present in the scene, but the agent is not penalized for hitting it.
    """

    def __init__(self, config) -> None:
        super(GoalLevel0, self).__init__(config=config)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]

        self._add_geoms(Goal(keepout=0.305, locations=[[1.1, 1.1]]))

        self._add_geoms(Hazards(size=0.7, num=1, keepout=0.705, locations=[[0, 0]]))

        self.last_dist_goal = None
