import numpy as np
from mushroom_rl.core import Core

from cremini_rl.algorithms.safe_layer_td3 import SafeLayerTD3, SafeLayerDDPG

class SafeCore(Core):
    def __init__(self, agent, mdp, callbacks_fit=None, callback_step=None, record_dictionary=None):
        super(SafeCore, self).__init__(agent, mdp, callbacks_fit, callback_step, record_dictionary)
        self._return_prev_cost = isinstance(agent, (SafeLayerTD3, SafeLayerDDPG))

    def reset(self, initial_states=None):
        super(SafeCore, self).reset(initial_states)
        self._cost = 0

    def _step(self, render, record):
        """
        Single step.

        Args:
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the agent, the reward obtained, the reached
            state, the absorbing flag of the reached state and the last step flag.

        """
        action = self.agent.draw_action(self._state)

        transformed_action = action
        if hasattr(self.agent, 'preprocess_action') and callable(self.agent.preprocess_action):
            transformed_action = self.agent.preprocess_action(self._state[np.newaxis], action, self._cost)

        next_state, reward, cost, absorbing, step_info = self.mdp.step(transformed_action)

        self._episode_steps += 1

        if render:
            frame = self.mdp.render(record)

            if record:
                self._record(frame)

        last = not (
            self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        prev_cost = self._cost
        self._cost = cost

        if self._return_prev_cost:
            return (state, action, reward, next_state, cost, prev_cost, absorbing, last), step_info

        return (state, action, reward, next_state, cost, absorbing, last), step_info
