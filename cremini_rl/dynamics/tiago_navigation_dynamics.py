from .dynamics import ControlAffineSystem
import numpy as np


class TiagoNavigationDynamics(ControlAffineSystem):
    # obs[0:2] = tiago_target
    # obs[2:9] = tiago_pose_vel
    # obs[9:16] = fetch_pose_vel
    # obs[16:18] = prev_action
    # obs[18:20] = finger_pos
    # obs[20:22] = finger_vel
    def __init__(self):
        super().__init__(3, 2, None, dim_x=5)

    def get_q(self, state):
        state = np.atleast_2d(state)
        q = np.hstack([state[:, 2].reshape(-1, 1), state[:, 3].reshape(-1, 1), np.arctan2(state[:, 5], state[:, 4]).reshape(-1, 1)])
        return q

    def get_x(self, state):
        state = np.atleast_2d(state)
        x = np.hstack([state[:, 9].reshape(-1, 1), state[:, 10].reshape(-1, 1), np.arctan2(state[:, 12], state[:, 11]).reshape(-1, 1),
                       state[:, 18].reshape(-1, 1), state[:, 19].reshape(-1, 1)])
        return x

    def get_x_dot(self, state):
        state = np.atleast_2d(state)
        return state[:, [13, 14, 15, 20, 21], np.newaxis]

    def G(self, q):
        """
        q = [x, y, theta]
        action = [v, omega]
        """
        N = len(q)
        G = np.zeros((N, self.dim_q, self.dim_u))

        G[:, 0, 0] = np.cos(q[:, 2])
        G[:, 1, 0] = np.sin(q[:, 2])
        G[:, 2, 1] = 1

        return G

    def f(self, q):
        N = len(q)
        return np.zeros((N, self.dim_q, 1))