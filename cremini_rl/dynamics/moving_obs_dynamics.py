import numpy as np
from .dynamics import VelocityControlSystem


class MovingObsDynamics(VelocityControlSystem):
    def __init__(self, dim_q, index_q, index_x, index_x_dot, vel_limit):
        super().__init__(dim_q, index_q, vel_limit)
        self.index_x = index_x
        self.index_x_dot = index_x_dot
        self.dim_x = len(index_x)
        self._add_save_attr(
            index_x='primitive',
            index_x_dot='primitive'
        )

    def get_x(self, state):
        state = np.atleast_2d(state)
        return state[:, self.index_x]

    def get_x_dot(self, state):
        state = np.atleast_2d(state)
        return state[:, self.index_x_dot, None]
