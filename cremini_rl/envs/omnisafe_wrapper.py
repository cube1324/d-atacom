from __future__ import annotations

import random
from typing import Any, ClassVar

import torch
import numpy as np

from gymnasium import spaces

import omnisafe
from omnisafe.envs.core import CMDP, env_register

from cremini_rl.envs.tiago_navigation_env import TiagoNavigationEnv
from cremini_rl.envs.cartpole_goal_env import SafeCartPoleEnv
from cremini_rl.envs.air_hockey_env import PlanarAirhockeyEnv


@env_register
class OmnisafeWrapper(CMDP):
    _support_envs: ClassVar[list[str]] = ['cartpole', 'tiago_navigation', 'docking2d', 'planar_air_hockey']
    # automatically reset when `terminated` or `truncated`
    need_auto_reset_wrapper = True
    # set `truncated=True` when the total steps exceed the time limit.
    need_time_limit_wrapper = False

    env_id_dict = {"cartpole": SafeCartPoleEnv, "tiago_navigation": TiagoNavigationEnv, "docking2d": DockingEnv,
                   "planar_air_hockey": PlanarAirhockeyEnv}

    def __init__(self, env_id: str, **kwargs: dict[str, Any]) -> None:
        self._count = 0
        self._num_envs = 1

        # self._device = kwargs['device']

        self._base_env = self.env_id_dict[env_id](return_cost=True)

        self._observation_space = spaces.Box(low=self._base_env.info.observation_space.low,
                                             high=self._base_env.info.observation_space.high)
        self._action_space = spaces.Box(low=self._base_env.info.action_space.low,
                                        high=self._base_env.info.action_space.high)

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1

        obs, reward, cost, terminated, info = self._base_env.step(action.cpu().numpy())

        obs = torch.as_tensor(obs).float()
        reward = torch.as_tensor(reward).float()
        cost = torch.as_tensor(np.maximum(cost, 0)).float()
        terminated = torch.as_tensor(terminated).float()
        truncated = torch.as_tensor(self._count > self.max_episode_steps)
        info["final_observation"] = obs

        return obs, reward, cost, terminated, truncated, info

    @property
    def max_episode_steps(self) -> int:
        """The max steps per episode."""
        return self._base_env.info.horizon

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        self.set_seed(seed)

        obs = self._base_env.reset()
        obs = torch.as_tensor(obs).float()

        self._count = 0
        return obs, {}

    def get_cost_from_obs_tensor(self, obs):
        cost = self._base_env.cost(obs.reshape(-1, 16).numpy())

        cost = torch.from_numpy(cost).float()

        return cost.reshape(obs.shape[:-1] + (1,))

    def set_seed(self, seed: int) -> None:
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def close(self) -> None:
        pass

    def render(self) -> Any:
        self._base_env.render()


if __name__ == "__main__":
    # agent = omnisafe.Agent(
    #     'PPOLag',
    #     'cartpole',
    # )
    # agent.learn()
    import bisect

    d = OmnisafeWrapper.precompute_constraints()

    a = KeyArray([0, 1, 0.1])

    res = bisect.bisect_left(list(d.keys()), a)
    print(res)
    print(list(d.keys())[res])

    # a = KeyArray([1, 2, 3])
    # b = KeyArray([1, 2, 1])
    # l = [a, b]
    # print(sorted(l))
    # print(a < 1)
