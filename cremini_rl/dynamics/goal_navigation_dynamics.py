from .dynamics import ControlAffineSystem
import numpy as np
import math

from numba import njit
import numba.np.extensions as ext

from functools import partial
import timeit


class Index(int):
    def __new__(cls, x, n=16):
        cls.n = n
        return super().__new__(cls, x)

    def __add__(self, other):
        if type(other) is np.ndarray:
            return other.__add__(self) % self.n

        res = super().__add__(other)
        return self.__class__(res % self.n, n=self.n)

    def __sub__(self, other):
        res = super().__sub__(other)
        return self.__class__(res % self.n, n=self.n)


class GoalNavigationControlSystem(ControlAffineSystem):
    """
    Observation:
    [0-3] linear acceleration
    [3-6] linear velocity
    [6-9] angular velocity
    [9-12] magnetometer???

    [12-28] goal lidar
    [28-44] hazards lidar
    [44-60] vases lidar


    State for dynamics:
    [0-16] hazards lidar
    [16-32] vases lidar
    """

    def __init__(self, vases=True):
        if vases:
            agent_idx = [*range(28, 60)]
            super().__init__(33, 2, agent_idx)

        else:
            agent_idx = [*range(28, 44)]
            super().__init__(17, 2, agent_idx)

        self.agent_idx = agent_idx
        self.vases = vases

        self.n_bins = 16

        self.delta_theta = 2 * np.pi / self.n_bins
        self.sin_delta_theta = math.sin(self.delta_theta)
        self.cos_delta_theta = math.cos(self.delta_theta)

        self.thetas = np.arange(0.5, self.n_bins, 1) * self.delta_theta  # Center of each bin of the lidar

        self.raw_points = np.array([[math.cos(self.thetas[i]), math.sin(self.thetas[i])] for i in range(self.n_bins)])

        # M = [[5.18879020e-03, 0.00000000e+00], [0.00000000e+00, 5.18879020e-03]]
        # for x and y
        self.m_inv = 1 / 5.18879020e-03

        self._add_save_attr(
            agent_idx='pickle',
            vases='primitive',
            n_bins='primitive',
            delta_theta='primitive',
            thetas='numpy',
            m_inv='primitive'
        )

    def get_q(self, state):
        state = np.atleast_2d(state)

        vel = np.linalg.norm(state[:, 3:4], axis=1, keepdims=True)
        return np.concatenate([state[:, self.agent_idx], vel], axis=1)

    def f(self, q):
        """
        State q:
        [0-16] hazards lidar
        [16-32] vases lidar
        [-1] velocity
        """
        N = len(q)

        f = np.zeros((N, self.dim_q, 1))

        f[:, 0:16, 0] = compute_linear_lidar_dot(q[:, 0:16], 1, self.n_bins, self.thetas,
                                                 self.raw_points) * q[:, -1, None]

        if self.vases:
            f[16:32, 0] = compute_linear_lidar_dot(q[:, 16:32], 1, self.n_bins, self.thetas,
                                                   self.raw_points) * q[:, -1, None]

        return f

    def G(self, q, alpha):
        """
        u = [x_dot, theta_dot]

        State q:
        [0-16] hazards lidar
        [16-32] vases lidar
        """
        N = len(q)

        G = np.zeros((N, self.dim_q, self.dim_u))

        G[:, 0:16, 1] = compute_angular_lidar_dot(q[:, 0:16], np.sign(alpha[:, 1]), self.n_bins, self.sin_delta_theta,
                                                  self.cos_delta_theta)

        if self.vases:
            G[:, 16:32, 1] = compute_angular_lidar_dot(q[:, 16:32], np.sign(alpha[:, 1]), self.n_bins,
                                                       self.sin_delta_theta, self.cos_delta_theta)

        G[:, -1, 0] = self.m_inv
        return G


@njit
def compute_linear_lidar_dot(lidar_obs, sign_x_dot, n_bins, thetas, raw_points):
    distance = _lidar_to_dist(lidar_obs)

    lidar_dot = np.zeros_like(distance)

    for i in range(n_bins):
        index_1 = (i - 1) % n_bins
        index_2 = i
        index_3 = (i + 1) % n_bins

        p_1 = raw_points[index_1] * distance[:, index_1, None]
        p_2 = raw_points[index_2] * distance[:, index_2, None]
        p_3 = raw_points[index_3] * distance[:, index_3, None]

        selected_idx = (_select_q(p_1, p_2, p_3, sign_x_dot) + index_2) % n_bins

        # selected_distance = distance[np.arange(distance.shape[0]), selected_idx]
        # Double list idx not supported in numba

        selected_distance = np.zeros(distance.shape[0])
        for j in range(distance.shape[0]):
            selected_distance[j] = distance[j, selected_idx[j]]

        q_dot_linear = _compute_linear_lidar_dot(distance[:, index_2],
                                                 selected_distance,
                                                 thetas[index_2], thetas[selected_idx])

        update_idx = distance[:, index_2] < 3

        lidar_dot[update_idx, index_2] = _dist_dot_to_lidar_dot(q_dot_linear[update_idx])
    return lidar_dot


@njit
def compute_angular_lidar_dot(lidar_obs, sign_theta_dot, n_bins, sin_delta_theta, cos_delta_theta):
    distance = _lidar_to_dist(lidar_obs)

    lidar_dot = np.zeros_like(distance)

    for i in range(n_bins):
        new_index = (sign_theta_dot.astype(np.int8) + i) % n_bins

        # selected_distance = distance[np.arange(distance.shape[0]), selected_idx]
        # Double list idx not supported in numba

        selected_distance = np.zeros(distance.shape[0])
        for j in range(distance.shape[0]):
            selected_distance[j] = distance[j, new_index[j]]

        q_dot_angular = sign_theta_dot * _compute_angular_lidar_dot(distance[:, i], selected_distance, sin_delta_theta,
                                                                    cos_delta_theta)

        update_idx = distance[:, i] < 3

        lidar_dot[update_idx, i] = _dist_dot_to_lidar_dot(q_dot_angular)[update_idx]

    return lidar_dot


@njit(inline='always')
def _lidar_to_dist(x):
    return - x * 3 + 3


@njit(inline='always')
def _dist_dot_to_lidar_dot(x_dot):
    return - x_dot / 3


@njit
def _select_q(p_1, p_2, p_3, sign_x_dot):
    direction = np.sign(sign_x_dot * p_2[:, 1]).astype(np.int8)
    temp = np.sign(ext.cross2d(p_1 - p_2, p_2)).astype(np.int8)

    # return -1 if equal, 1 otherwise
    return - direction * temp


@njit
def _compute_linear_lidar_dot(q1, q2, theta_1, theta_2):
    # q_dot = - (np.sin(theta_1) * q1 - np.sin(theta_2) * q2) / (
    #         np.sin(theta_1 - theta_2) * q2)  # * v added in dynamics
    q_dot = - (math.sin(theta_1) * q1 - np.sin(theta_2) * q2) / (
            np.sin(theta_1 - theta_2) * q2)  # * v added in dynamics

    return q_dot


@njit
def _compute_angular_lidar_dot(q1, q2, sin_delta_theta, cos_delta_theta):
    q_dot = - q1 / sin_delta_theta * (
            q1 / q2 - cos_delta_theta)  # * omega is added in dynamics

    return q_dot


class GoalNavigationControlSystemDeprecated(ControlAffineSystem):
    """
    Observation:
    [0-3] linear acceleration
    [3-6] linear velocity
    [6-9] angular velocity
    [9-12] magnetometer???

    [12-28] goal lidar
    [28-44] hazards lidar
    [44-60] vases lidar


    State for dynamics:
    [0-16] hazards lidar
    [16-32] vases lidar
    """

    def __init__(self, vases=True):
        if vases:
            agent_idx = [*range(28, 60)]
            super().__init__(33, 2, agent_idx)

        else:
            agent_idx = [*range(28, 44)]
            super().__init__(17, 2, agent_idx)

        self.agent_idx = agent_idx
        self.vases = vases

        self.n_bins = 16

        self.delta_theta = 2 * np.pi / self.n_bins

        self.thetas = np.arange(0.5, self.n_bins, 1) * self.delta_theta  # Center of each bin of the lidar

        # M = [[5.18879020e-03, 0.00000000e+00], [0.00000000e+00, 5.18879020e-03]]
        # for x and y
        self.m_inv = 1 / 5.18879020e-03

        self._add_save_attr(
            agent_idx='pickle',
            vases='primitive',
            n_bins='primitive',
            delta_theta='primitive',
            thetas='numpy',
            m_inv='primitive'
        )

    def get_q(self, state):
        state = np.atleast_2d(state)

        vel = np.linalg.norm(state[:, 3:4], axis=1, keepdims=True)
        return np.concatenate([state[:, self.agent_idx], vel], axis=1)

    def f(self, batch_q):
        N = len(batch_q)
        f = np.zeros((N, self.dim_q, 1))
        for i, q in enumerate(batch_q):
            f[i] = self._f(q)

        return f

    def _f(self, q):
        """
        State q:
        [0-16] hazards lidar
        [16-32] vases lidar
        [-1] velocity
        """
        f = np.zeros((self.dim_q, 1))

        f[0:16, 0] = self.compute_linear_lidar_dot(q[0:16], 1) * q[-1]

        if self.vases:
            f[16:32, 0] = self.compute_linear_lidar_dot(q[16:32], 1) * q[-1]

        # self.data.qvel[:2] = 100
        # self.data.qvel[2] = 100
        # self.data.qacc[:3] = 0
        # # mujoco.mj_forward(self.model, self.data)
        # # full_m = np.zeros((self.model.nv, self.model.nv))
        # # mujoco.mj_fullM(self.model, full_m, M=self.data.qM)
        # # res = np.zeros(3)
        # # mujoco.mj_rne(self.model, self.data, 0, res)
        # # print(res)
        #
        # mujoco.mj_forward(self.model, self.data)
        # print(self.data.qfrc_bias)
        return f

    def G(self, batch_q, alpha):
        N = len(batch_q)
        G = np.zeros((N, self.dim_q, self.dim_u))
        for i, q in enumerate(batch_q):
            G[i] = self._G(q, alpha[i])

        return G

    def _G(self, q, alpha):
        """
        u = [x_dot, theta_dot]

        State q:
        [0-16] hazards lidar
        [16-32] vases lidar
        """

        G = np.zeros((self.dim_q, self.dim_u))

        # G[0:16, 0] = self.compute_linear_lidar_dot(q[0:16], np.sign(alpha[0]))
        G[0:16, 1] = self.compute_angular_lidar_dot(q[0:16], np.sign(alpha[1]))

        if self.vases:
            # G[16:32, 0] = self.compute_linear_lidar_dot(q[16:32], np.sign(alpha[0]))
            G[16:32, 1] = self.compute_angular_lidar_dot(q[16:32], np.sign(alpha[1]))

        G[-1, 0] = self.m_inv
        return G

    def compute_linear_lidar_dot(self, lidar_obs, sign_x_dot):
        distance = self._lidar_to_dist(lidar_obs)
        # distance = lidar_obs

        lidar_dot = np.zeros_like(distance)

        for i in range(self.n_bins):
            if distance[i] < 3:
                index = Index(i, self.n_bins)

                q_1 = distance[index - 1]
                q_2 = distance[index]
                q_3 = distance[index + 1]

                p_1 = self._get_p(q_1, index - 1)
                p_2 = self._get_p(q_2, index)
                p_3 = self._get_p(q_3, index + 1)

                v = np.array([sign_x_dot, 0])

                selected_idx = index + self._select_q(p_1, p_2, p_3, v)

                q_dot_linear = self._compute_linear_lidar_dot(distance[index], distance[selected_idx],
                                                              self.thetas[index], self.thetas[selected_idx])

                lidar_dot[index] = self._dist_dot_to_lidar_dot(q_dot_linear)
                # lidar_dot[index] = q_dot_linear

        return lidar_dot

    def compute_angular_lidar_dot(self, lidar_obs, sign_theta_dot):
        distance = self._lidar_to_dist(lidar_obs)
        # distance = lidar_obs

        lidar_dot = np.zeros_like(distance)

        for i in range(self.n_bins):
            if distance[i] < 3:
                index = Index(i, self.n_bins)

                if sign_theta_dot == -1:
                    # if theta_dot negative use right lidar and turn it positive
                    q_dot_angular = -self._compute_angular_lidar_dot(distance[index], distance[index - 1])
                else:
                    q_dot_angular = self._compute_angular_lidar_dot(distance[index], distance[index + 1])

                lidar_dot[index] = self._dist_dot_to_lidar_dot(q_dot_angular)
                # lidar_dot[index] = q_dot_angular

        return lidar_dot

    def _get_p(self, q, i):
        return np.array([np.cos(self.thetas[i]), np.sin(self.thetas[i])]) * q

    def _lidar_to_dist(self, x):
        return - x * 3 + 3

    def _dist_dot_to_lidar_dot(self, x_dot):
        return - x_dot / 3

    def _select_q(self, p_1, p_2, p_3, v):
        direction = np.sign(np.cross(v, p_2))

        if np.sign(np.cross(p_1 - p_2, p_2)) == direction:
            return -1
        return 1

    def _compute_linear_lidar(self, q1, q2, theta_1, theta_2, delta_x):
        return q1 - (np.sin(theta_1) * q1 - np.sin(theta_2) * q2) / (np.sin(theta_1 - theta_2) * q2) * delta_x

    def _compute_linear_lidar_dot(self, q1, q2, theta_1, theta_2):
        q_dot = - (np.sin(theta_1) * q1 - np.sin(theta_2) * q2) / (
                np.sin(theta_1 - theta_2) * q2)  # * v added in dynamics

        return q_dot

    def _compute_angular_lidar(self, q1, q2, delta_theta, theta):
        return np.sin(delta_theta) * q1 / (np.sin(delta_theta - theta) + np.sin(theta) * q1 / q2)

    def _compute_angular_lidar_dot(self, q1, q2):
        q_dot = - q1 / np.sin(self.delta_theta) * (q1 / q2 - np.cos(self.delta_theta))  # * omega is added in dynamics

        return q_dot


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from matplotlib.patches import Polygon


    def plot_lidar(ax, lidar, center=np.array([0, 0]), offset_theta=0, color="g"):
        thetas = np.pi * 2 * np.arange(0.5, 16, 1) / 16

        # points = np.vstack([np.cos(thetas), np.sin(thetas)]).reshape(16, 2) * thetas.reshape(-1, 1)
        points = [[np.cos(offset_theta + theta) * dist, np.sin(offset_theta + theta) * dist] for theta, dist in
                  zip(thetas, lidar)]

        points = np.array(points) + center
        p_old = Polygon(points, closed=True, fill=False, edgecolor=color)

        ax.add_patch(p_old)

        ax.scatter(*center, c=color)


    def tes_lidar_dot():
        lidar = np.ones(16) * 0.5

        pos = np.zeros(2)
        vel = 1
        omega = 0  # -45 * np.pi / 180
        action = np.array([vel, omega])
        dt = 0.1

        lidar[3] += 0.2
        lidar[4] += 0.2

        dyn = GoalNavigationControlSystem()

        G = dyn._G(np.concatenate([lidar, lidar]), action)
        lidar_dot = G[:16] @ action

        # lidar_linear_dot = dyn.compute_linear_lidar_dot(lidar, np.sign(vel)) * vel
        # lidar_angular_dot = dyn.compute_angular_lidar_dot(lidar, np.sign(omega)) * omega
        #
        # lidar_dot = lidar_angular_dot + lidar_linear_dot

        lidar_new = lidar + lidar_dot * dt

        pos_new = pos + np.array([vel, 0]) * dt
        theta_new = omega * dt

        fig, ax = plt.subplots()

        # plot_lidar(ax, lidar, pos, 0, "b")
        # plot_lidar(ax, lidar_new, pos_new, theta_new)

        plot_lidar(ax, dyn._lidar_to_dist(lidar), pos, 0, "b")
        plot_lidar(ax, dyn._lidar_to_dist(lidar_new), pos_new, theta_new)

        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])

        plt.show()


    def linear_simple():
        q1 = np.array([1, 1])
        q2 = np.array([1, 0])
        q3 = np.array([0, 1])

        dyn = GoalNavigationControlSystem(None, None)

        res = dyn._compute_linear_lidar(np.sqrt(2), 1, 45 * np.pi / 180, 0, 0.25)

        print(res)


    def angular_simple():
        dyn = GoalNavigationControlSystem(None, None)

        omega = 45 * np.pi / 180  # rad / s
        dt = 1
        q_1 = 1
        q_2 = np.sqrt(2)

        q_1_dot = dyn._compute_angular_lidar_dot(q_1, q_2, omega)  # m / s
        print(q_1_dot)
        q_1_new = q_1 + q_1_dot * dt

        print(q_1_new)


    # linear_simple()

    # angular_simple()

    # tes_lidar_dot()

    rng = np.random.RandomState(1234567890)

    q = rng.normal(size=(256, 17))
    a = rng.normal(size=(256, 2))

    dyn_baselines = GoalNavigationControlSystemDeprecated(vases=False)

    dyn = GoalNavigationControlSystem(vases=False)

    f = dyn.f(q)
    f_baseline = dyn_baselines.f(q)

    assert np.allclose(f_baseline, f)

    G = dyn.G(q, a)
    G_baseline = dyn_baselines.G(q, a)

    assert np.allclose(G_baseline, G)

    func = partial(dyn.f, q)

    times = timeit.repeat("func()", setup="from __main__ import func", repeat=50, number=1)

    print(np.mean(times), np.std(times), times)

    # baseline: 0.40334906000000004 0.03074511630068131
    # no np.cross: 0.1287822799999999 0.0005993982629272482
    # jit: 0.0002257299999999418 1.1769796089988815e-05

    # Laptop:
    # baseline: 0.10101754000000004 0.0018950755410802537
    # no batch loop: 0.001784339999999851 0.0003721575451340859
