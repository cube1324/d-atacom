from .dynamics import ControlAffineSystem
import numpy as np


class CartPoleControlSystem(ControlAffineSystem):
    def __init__(self,  mc, mp, l, g, u_limit, Jp=0):
        super().__init__(4, 1, None)

        self.nominal_dyn = {'mc': mc, 'mp': mp, 'l': l, 'Jp': Jp}
        self._mc = mc
        self._mp = mp
        self._l = l
        self._g = g
        self._Jp = Jp
        self.u_limit = u_limit

        self._add_save_attr(
            _mc='primitive',
            _mp='primitive',
            _l='primitive',
            _g='primitive',
            _Jp='primitive',
            u_limit='primitive',
            nominal_dyn='pickle'
        )

    def get_q(self, state):
        state = np.atleast_2d(state)
        theta = np.arctan2(state[:, 1], state[:, 2])

        return np.hstack([state[:, 0].reshape(-1, 1), theta.reshape(-1, 1), state[:, 3:]])

    def f(self, q):
        # Follow the derivation: https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
        dd_x_theta = self._inv_inertia(q) @ self._drift(q)
        N = len(q)

        f = np.zeros((N, 4, 1))

        f[:, :2] = q[:, 2:, np.newaxis]
        f[:, 2:] = dd_x_theta

        return f

    def G(self, q):
        N = len(q)

        inv_inertia = self._inv_inertia(q)

        G = np.zeros((N, 4, 1))
        G[:, 2:] = self.u_limit * inv_inertia[:, :, :1]
        return G

    def _drift(self, q):
        N = len(q)

        dtheta = q[:, 3]
        sin_theta = np.sin(q[:, 1]).reshape(-1, 1)

        # np.array([self._mp * self._l * dtheta ** 2 * sin_theta,
        #           self._mp * self._g * self._l * sin_theta])

        drift = np.array([self._mp * self._l,
                         self._mp * self._g * self._l])

        drift_stacked = np.repeat(drift[np.newaxis, :], N, axis=0)

        drift_stacked[:, 0] *= dtheta**2
        drift_stacked *= sin_theta
        return drift_stacked.reshape(N, 2, 1)

    def _inv_inertia(self, q):
        N = len(q)
        cos_theta = np.cos(q[:, 1])
        # inertia = np.array([[self._mc + self._mp, self._mp * self._l * cos_theta],
        #                     [self._mp * self._l * cos_theta, self._mp * self._l ** 2 + self._Jp]])

        inertia = np.array([[self._mc + self._mp, self._mp * self._l],
                            [self._mp * self._l, self._mp * self._l ** 2 + self._Jp]])

        stacked_inertia = np.repeat(inertia[np.newaxis, :, :], N, axis=0)

        stacked_inertia[:, 0, 1] *= cos_theta
        stacked_inertia[:, 1, 0] *= cos_theta

        inv_inertia = np.linalg.inv(stacked_inertia)
        return inv_inertia