import numpy as np

from mushroom_rl.core import Serializable

from mushroom_rl.utils.replay_memory import ReplayMemory



class SafeReplayMemory(ReplayMemory):
    def __init__(self, initial_size, max_size):
        super().__init__(initial_size, max_size)

        self._add_save_attr(
            _cost='pickle!',
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
        """

        for el in dataset:
            self._states[self._idx] = el[0]
            self._actions[self._idx] = el[1]
            self._rewards[self._idx] = el[2]
            self._next_states[self._idx] = el[3]
            self._cost[self._idx] = el[4]
            self._absorbing[self._idx] = el[5]
            self._last[self._idx] = el[6]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        a = list()
        r = list()
        ss = list()
        c = list()
        ab = list()
        last = list()
        for i in np.random.randint(self.size, size=n_samples):
            s.append(np.array(self._states[i]))
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(np.array(self._next_states[i]))
            c.append(self._cost[i])
            ab.append(self._absorbing[i])
            last.append(self._last[i])

        return np.array(s), np.array(a), np.array(r), np.array(ss), \
            np.array(c), np.array(ab), np.array(last)

    def reset(self):
        super(SafeReplayMemory, self).reset()

        self._cost = [None for _ in range(self._max_size)]


class SafeLayerReplayMemory(ReplayMemory):
    def __init__(self, initial_size, max_size):
        super(SafeLayerReplayMemory, self).__init__(initial_size, max_size)

        self._add_save_attr(
            _cost='pickle!',
            _prev_cost='pickle!'
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
        """

        for el in dataset:
            self._states[self._idx] = el[0]
            self._actions[self._idx] = el[1]
            self._rewards[self._idx] = el[2]
            self._next_states[self._idx] = el[3]
            self._cost[self._idx] = el[4]
            self._prev_cost[self._idx] = el[5]
            self._absorbing[self._idx] = el[6]
            self._last[self._idx] = el[7]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0


    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        a = list()
        r = list()
        ss = list()
        c = list()
        pc = list()
        ab = list()
        last = list()
        for i in np.random.randint(self.size, size=n_samples):
            s.append(np.array(self._states[i]))
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(np.array(self._next_states[i]))
            c.append(self._cost[i])
            pc.append(self._prev_cost[i])
            ab.append(self._absorbing[i])
            last.append(self._last[i])

        return np.array(s), np.array(a), np.array(r), np.array(ss),\
            np.array(c), np.array(pc), np.array(ab), np.array(last)

    def reset(self):
        super(SafeLayerReplayMemory, self).reset()

        self._cost = [None for _ in range(self._max_size)]
        self._prev_cost = [None for _ in range(self._max_size)]



class ConstraintReplayMemory(Serializable):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, initial_size, max_size):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory
                can contain.

        """
        self._initial_size = initial_size
        self._max_size = max_size

        self.reset()

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _idx='primitive',
            _full='primitive',
            _states='pickle!',
            _actions='pickle!',
            _constraints='pickle!',
            _log_prob='pickle!',
            _next_states='pickle!',
            _absorbing='pickle!',
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.
        """

        for el in dataset:
            self._states[self._idx] = el[0]
            self._actions[self._idx] = el[1]
            self._next_states[self._idx] = el[3]
            self._constraints[self._idx] = el[4]
            self._log_prob[self._idx] = el[5]
            self._absorbing[self._idx] = el[6]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        a = list()
        c = list()
        p = list()
        ss = list()
        ab = list()
        for i in np.random.randint(self.size, size=n_samples):
            s.append(np.array(self._states[i]))
            a.append(np.array(self._actions[i]))
            c.append(self._constraints[i])
            p.append(self._log_prob[i])
            ss.append(np.array(self._next_states[i]))
            ab.append(self._absorbing[i])

        return np.array(s), np.array(a), np.array(c), p, np.array(ss), np.array(ab)

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = [None for _ in range(self._max_size)]
        self._actions = [None for _ in range(self._max_size)]
        self._log_prob = [None for _ in range(self._max_size)]
        self._constraints = [None for _ in range(self._max_size)]
        self._next_states = [None for _ in range(self._max_size)]
        self._absorbing = [None for _ in range(self._max_size)]

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size > self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size

    def _post_load(self):
        if self._full is None:
            self.reset()


class ConstraintBucketReplayMemory(Serializable):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, initial_size, max_size, num_bucket=20, min=None, max=None):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory
                can contain.

        """

        self._min = min
        self._max = max
        self._num_buckets = num_bucket

        self._buckets = None

        if self._num_buckets == 2:
            self._buckets = [0]

        self._initial_size = initial_size
        self._max_size = max_size

        self._max_size_per_bucket = self._max_size // self._num_buckets

        self.reset()

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _max_size_per_bucket='primitive',
            _idx='pickle',
            _full='pickle',
            _buckets='pickle',
            _states='pickle!',
            _actions='pickle!',
            _log_prob='pickle!',
            _constraints='pickle!',
            _next_states='pickle!',
            _absorbing='pickle!',
            _num_buckets='primitive',
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.
        """
        cost = [el[4] for el in dataset]

        if self._buckets is None:
            if self._min is None:
                self._min = min(cost)
            if self._max is None:
                self._max = max(cost)

            # self._buckets = np.linspace(self._min, self._max, self._num_buckets - 1)
            cost_range = self._max - self._min
            bucket_size = cost_range / self._num_buckets

            self._buckets = [self._min + i * bucket_size for i in range(1, self._num_buckets)]

        buckets = np.digitize(cost, bins=self._buckets)

        for el, b in zip(dataset, buckets):
            self._states[b][self._idx[b]] = el[0]
            self._actions[b][self._idx[b]] = el[1]
            self._next_states[b][self._idx[b]] = el[3]
            self._constraints[b][self._idx[b]] = el[4]
            self._log_prob[b][self._idx[b]] = el[5]
            self._absorbing[b][self._idx[b]] = el[6]

            self._idx[b] += 1
            if self._idx[b] == self._max_size_per_bucket:
                self._full[b] = True
                self._idx[b] = 0

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        cum_size = np.cumsum(self.size)

        s = list()
        a = list()
        c = list()
        p = list()
        ss = list()
        ab = list()

        for i in np.random.randint(sum(self.size), size=n_samples):
            b = 0

            while i >= cum_size[b]:
                b += 1

            index = i
            if b > 0:
                index -= cum_size[b - 1]

            s.append(np.array(self._states[b][index]))
            a.append(np.array(self._actions[b][index]))
            c.append(self._constraints[b][index])
            p.append(self._log_prob[b][index])
            ss.append(np.array(self._next_states[b][index]))
            ab.append(self._absorbing[b][index])

        s = np.array(s)
        a = np.array(a)
        c = np.array(c)
        # p = np.array(p)
        ss = np.array(ss)
        ab = np.array(ab)

        return s, a, c, p, ss, ab

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = [0 for _ in range(self._num_buckets)]
        self._full = [False for _ in range(self._num_buckets)]
        self._states = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._actions = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._log_prob = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._constraints = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._next_states = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._absorbing = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return sum(self.size) > self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return [self._idx[b] if not self._full[b] else self._max_size_per_bucket for b in range(self._num_buckets)]

    def _post_load(self):
        if self._full is None:
            self.reset()


class GaussianAtacomReplayMemory(Serializable):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, initial_size, max_size, num_bucket=20, min=None, max=None):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory
                can contain.

        """

        self._min = min
        self._max = max
        self._num_buckets = num_bucket

        self._buckets = None

        if self._num_buckets == 2:
            self._buckets = [0]

        self._initial_size = initial_size
        self._max_size = max_size

        self._max_size_per_bucket = self._max_size // self._num_buckets

        self.reset()

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _max_size_per_bucket='primitive',
            _idx='pickle',
            _full='pickle',
            _buckets='pickle',
            _states='pickle!',
            _actions='pickle!',
            _unclipped_action='pickle!',
            _log_prob='pickle!',
            _rewards='pickle!',
            _constraints='pickle!',
            _next_states='pickle!',
            _absorbing='pickle!',
            _last='pickle!',
            _num_buckets='primitive',
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.
        """
        cost = [el[4] for el in dataset]

        if self._buckets is None:
            if self._min is None:
                self._min = min(cost)
            if self._max is None:
                self._max = max(cost)

            # self._buckets = np.linspace(self._min, self._max, self._num_buckets - 1)
            cost_range = self._max - self._min
            bucket_size = cost_range / self._num_buckets

            self._buckets = [self._min + i * bucket_size for i in range(1, self._num_buckets)]

        buckets = np.digitize(cost, bins=self._buckets)

        for el, b in zip(dataset, buckets):
            self._states[b][self._idx[b]] = el[0]
            self._actions[b][self._idx[b]] = el[1]
            self._rewards[b][self._idx[b]] = el[2]
            self._next_states[b][self._idx[b]] = el[3]
            self._constraints[b][self._idx[b]] = el[4]
            self._log_prob[b][self._idx[b]] = el[5][0]
            self._unclipped_action[b][self._idx[b]] = el[5][1]
            self._absorbing[b][self._idx[b]] = el[6]
            self._last[b][self._idx[b]] = el[7]

            self._idx[b] += 1
            if self._idx[b] == self._max_size_per_bucket:
                self._full[b] = True
                self._idx[b] = 0

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        cum_size = np.cumsum(self.size)

        s = list()
        a = list()
        u = list()
        p = list()
        r = list()
        c = list()
        ss = list()
        ab = list()
        l = list()

        for i in np.random.randint(sum(self.size), size=n_samples):
            b = 0

            while i >= cum_size[b]:
                b += 1

            index = i
            if b > 0:
                index -= cum_size[b - 1]

            s.append(np.array(self._states[b][index]))
            a.append(np.array(self._actions[b][index]))
            u.append(np.array(self._unclipped_action[b][index]))
            p.append(np.array(self._log_prob[b][index]))
            r.append(self._rewards[b][index])
            c.append(self._constraints[b][index])
            ss.append(np.array(self._next_states[b][index]))
            ab.append(self._absorbing[b][index])
            l.append(self._last[b][index])

        s = np.array(s)
        a = np.array(a)
        u = np.array(u)
        p = np.array(p)
        r = np.array(r)
        c = np.array(c)
        ss = np.array(ss)
        ab = np.array(ab)
        l = np.array(l)

        return s, a, u, p, r, c, ss, ab, l

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = [0 for _ in range(self._num_buckets)]
        self._full = [False for _ in range(self._num_buckets)]
        self._states = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._actions = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._unclipped_action = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._log_prob = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._rewards = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._constraints = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._next_states = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._absorbing = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]
        self._last = [[None for _ in range(self._max_size_per_bucket)] for _ in range(self._num_buckets)]

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return sum(self.size) > self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return [self._idx[b] if not self._full[b] else self._max_size_per_bucket for b in range(self._num_buckets)]

    def _post_load(self):
        if self._full is None:
            self.reset()


class BernoulliReplayMemory(Serializable):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, initial_size, max_size):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory
                can contain.

        """
        self._initial_size = initial_size
        self._max_size = max_size

        self.num_classes = {0: 0, 1: 0}

        self.reset()

        self.desired_ratio = 0.5

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _idx='primitive',
            _full='primitive',
            _states='pickle!',
            _constraints='pickle!',
            _next_states='pickle!',
            _absorbing='pickle!',
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.
        """

        for el in dataset:
            if self._full:
                ratio = self.num_classes[1] / self.size

                if ratio < self.desired_ratio:
                    while self._constraints[self._idx] == 1:
                        self._idx += 1
                        if self._idx == self._max_size:
                            self._idx = 0

                else:
                    while self._constraints[self._idx] == 0:
                        self._idx += 1
                        if self._idx == self._max_size:
                            self._idx = 0

                self.num_classes[self._constraints[self._idx]] -= 1

            self._states[self._idx] = el[0]
            self._next_states[self._idx] = el[3]
            self._constraints[self._idx] = el[4]
            self._absorbing[self._idx] = el[5]

            self.num_classes[el[4]] += 1

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        c = list()
        ss = list()
        ab = list()
        for i in np.random.randint(self.size, size=n_samples):
            s.append(np.array(self._states[i]))
            c.append(self._constraints[i])
            ss.append(np.array(self._next_states[i]))
            ab.append(self._absorbing[i])

        return np.array(s), np.array(c), np.array(ss), np.array(ab)

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = [None for _ in range(self._max_size)]
        self._constraints = [None for _ in range(self._max_size)]
        self._next_states = [None for _ in range(self._max_size)]
        self._absorbing = [None for _ in range(self._max_size)]

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size > self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size

    def _post_load(self):
        if self._full is None:
            self.reset()


