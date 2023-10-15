from abc import abstractmethod

import numpy as np

from mushroom_rl.core import Serializable


class ControlAffineSystem(Serializable):
    def __init__(self, dim_q, dim_u, index_q, dim_x=0, index_x=None):
        if index_x is None:
            index_x = []

        self.dim_q = dim_q
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.index_q = index_q
        self.index_x = index_x

        self._add_save_attr(
            dt='primitive',
            dim_q='primitive',
            dim_u='primitive',
            dim_x='primitive',
            index_q='pickle',
            index_x='pickle',
        )

    def get_q(self, state):
        state = np.atleast_2d(state)
        return state[:, self.index_q]

    def get_x(self, state):
        state = np.atleast_2d(state)
        return state[:, self.index_x]

    def get_x_dot(self, state):
        state = np.atleast_2d(state)

        N = len(state)
        x_dot = np.zeros((N, self.dim_x, 1))
        return x_dot

    @abstractmethod
    def f(self, q):
        pass

    @abstractmethod
    def G(self, q):
        pass

    def dq(self, q, u):
        assert u.shape[-1] == self.dim_u
        return self.f(q) + self.G(q) @ u


class LinearControlSystem(ControlAffineSystem):
    def __init__(self, dim_q, index_q, A, B, **kwargs):
        super().__init__(2 * dim_q, dim_q, index_q, **kwargs)

        self.A = A
        self.B = B

        self._add_save_attr(
            A='pickle',
            B='pickle'
        )

    def f(self, q):
        q = q.copy()
        q[:, :2] *= 100
        q[:, 2:] *= 0.5
        return self.A @ q[:, :, np.newaxis]

    def G(self, q):
        return self.B


class VelocityControlSystem(ControlAffineSystem):
    def __init__(self, dim_q, index_q, vel_limit):
        if np.isscalar(vel_limit):
            self.vel_limit = np.ones(dim_q) * vel_limit
        else:
            self.vel_limit = vel_limit

        self._add_save_attr(
            vel_limit='primitive',
        )
        super().__init__(dim_q, dim_q, index_q)

    def f(self, q):
        assert q.shape[-1] == self.dim_q
        return np.zeros((q.shape[0], self.dim_q, 1))

    def G(self, q):
        assert q.shape[-1] == self.dim_q
        return np.diag(self.vel_limit)


class AccelerationControlSystem(ControlAffineSystem):
    """
    We assume the q_hat is ordered by [q, q_dot], we therefore have
    """

    def __init__(self, dim_q, index_q, acc_limit):
        if np.isscalar(acc_limit):
            self.acc_limit = np.ones(dim_q) * acc_limit
        else:
            self.acc_limit = acc_limit

        self._add_save_attr(
            acc_limit='primitive',
        )
        super().__init__(dim_q * 2, dim_q, index_q)

    def f(self, q_hat):
        assert q_hat.shape[-1] == self.dim_q
        n_dim = int(self.dim_q / 2)

        f = np.zeros((q_hat.shape[0], self.dim_q, 1))
        f[:, :n_dim] = q_hat[:, n_dim:, np.newaxis]
        return f

    def G(self, q_hat):
        assert q_hat.shape[-1] == self.dim_q
        n_dim = int(self.dim_q / 2)
        return np.vstack([np.zeros((n_dim, n_dim)), np.diag(self.acc_limit)])

