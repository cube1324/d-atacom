from safety_gymnasium.tasks.goal.goal_level0 import GoalLevel0
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.free_geoms import Vases


class DenseGoalLevel1(GoalLevel0):
    """An agent must navigate to a goal while avoiding hazards.

    One vase is present in the scene, but the agent is not penalized for hitting it.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]

        self._add_geoms(Hazards(num=8, keepout=0.18))
        self._add_free_geoms(Vases(num=1, is_constrained=False))

        self.cost_conf.constrain_indicator = False

