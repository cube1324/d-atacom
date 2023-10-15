from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

ActivationLayer = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'selu': nn.SELU,
    'leaky_relu': nn.LeakyReLU,
    'softplus': nn.Softplus
}


def build_mlp(input_dim, n_features, output_dim, activation, output_mod=None):
    layers = [nn.Linear(input_dim, n_features[0]), ActivationLayer[activation]()]
    for i in range(1, len(n_features)):
        layers += [nn.Linear(n_features[i - 1], n_features[i]), ActivationLayer[activation]()]
    layers.append(nn.Linear(n_features[-1], output_dim))

    if output_mod is not None:
        layers.append(output_mod)
    trunk = nn.Sequential(*layers)
    return trunk


def weight_init(m, activation, custom_scale=1):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation) / custom_scale)


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, activation='relu', **kwargs):
        super(MLP, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.trunk = build_mlp(n_input, n_features, n_output, activation) #, output_mod=nn.Softplus())

        self.apply(partial(weight_init, activation=activation, custom_scale=5))

        self.trunk[-1].bias.data.fill_(-1)

    def forward(self, state):
        return self.trunk(torch.squeeze(state, 1).float()).squeeze()


class TD3CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        dim_action = kwargs['action_shape'][0]
        dim_state = n_input - dim_action
        n_output = output_shape[0]

        self._action_scaling = torch.tensor(kwargs['action_scaling'], dtype=torch.float32).to(
            device=torch.device('cuda' if kwargs['use_cuda'] else 'cpu'))

        # Assume there are two hidden layers
        assert len(n_features) == 2, 'TD3 critic needs 2 hidden layers'

        self._h1 = nn.Linear(dim_state + dim_action, n_features[0])
        self._h2_s = nn.Linear(n_features[0], n_features[1])
        self._h2_a = nn.Linear(dim_action, n_features[1], bias=False)
        self._h3 = nn.Linear(n_features[1], n_output)

        fan_in_h1, _ = nn.init._calculate_fan_in_and_fan_out(self._h1.weight)
        nn.init.uniform_(self._h1.weight, a=-1 / np.sqrt(fan_in_h1), b=1 / np.sqrt(fan_in_h1))

        fan_in_h2_s, _ = nn.init._calculate_fan_in_and_fan_out(self._h2_s.weight)
        nn.init.uniform_(self._h2_s.weight, a=-1 / np.sqrt(fan_in_h2_s), b=1 / np.sqrt(fan_in_h2_s))

        fan_in_h2_a, _ = nn.init._calculate_fan_in_and_fan_out(self._h2_a.weight)
        nn.init.uniform_(self._h2_a.weight, a=-1 / np.sqrt(fan_in_h2_a), b=1 / np.sqrt(fan_in_h2_a))

        nn.init.uniform_(self._h3.weight, a=-3e-3, b=3e-3)

    def forward(self, state, action):
        state = state.float()
        action = action.float() / self._action_scaling
        state_action = torch.cat((state, action), dim=1)

        features1 = F.relu(self._h1(state_action))
        features2_s = self._h2_s(features1)
        features2_a = self._h2_a(action)
        features2 = F.relu(features2_s + features2_a)

        q = self._h3(features2)
        return torch.squeeze(q)


class TD3ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        dim_state = input_shape[0]
        dim_action = output_shape[0]

        self._action_scaling = torch.tensor(kwargs['action_scaling']).to(
            device=torch.device('cuda' if kwargs['use_cuda'] else 'cpu'))

        # Assume there are two hidden layers
        assert len(n_features) == 2, 'TD3 actor needs two hidden layers'

        self._h1 = nn.Linear(dim_state, n_features[0])
        self._h2 = nn.Linear(n_features[0], n_features[1])
        self._h3 = nn.Linear(n_features[1], dim_action)

        fan_in_h1, _ = nn.init._calculate_fan_in_and_fan_out(self._h1.weight)
        nn.init.uniform_(self._h1.weight, a=-1 / np.sqrt(fan_in_h1), b=1 / np.sqrt(fan_in_h1))

        fan_in_h2, _ = nn.init._calculate_fan_in_and_fan_out(self._h2.weight)
        nn.init.uniform_(self._h2.weight, a=-1 / np.sqrt(fan_in_h2), b=1 / np.sqrt(fan_in_h2))

        nn.init.uniform_(self._h3.weight, a=-3e-3, b=3e-3)

    def forward(self, state):
        state = state.float()

        features1 = F.relu(self._h1(state))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        a = self._action_scaling * torch.tanh(a)

        return a


class GaussianConstraintNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, activation='relu', **kwargs):
        super(GaussianConstraintNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._mu_net = build_mlp(n_input, n_features, n_output, activation, output_mod=nn.Softplus())
        self._sigma_net = build_mlp(n_input, n_features, n_output, activation)

        self.apply(partial(weight_init, activation=activation))

    def forward(self, state):
        mu = self._mu_net(state.float())
        sigma = self._sigma_net(state.float())

        return mu.flatten(), sigma.flatten()


class SACCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, activation='relu', **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.trunk = build_mlp(n_input, n_features, n_output, activation)

        self.apply(partial(weight_init, activation=activation))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        return self.trunk(state_action).squeeze()


class SACActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, activation='relu', **kwargs):
        super(SACActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.trunk = build_mlp(n_input, n_features, n_output, activation)

    def forward(self, state):
        # IMPORTANT DO NOT SQUEEZE, mushroom breaks otherwise
        return self.trunk(torch.squeeze(state, 1).float())


class GaussianConstraintQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, activation='relu', **kwargs):
        super(GaussianConstraintQNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._mu_net = build_mlp(n_input, n_features, n_output, activation)
        self._sigma_net = build_mlp(n_input, n_features, n_output, activation, nn.Softplus())

        self.apply(partial(weight_init, activation=activation))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        mu = self._mu_net(state_action)
        sigma = self._sigma_net(state_action)

        return mu.squeeze(), sigma.squeeze()


class ImplicitQuantileConstraint(nn.Module):
    def __init__(self, input_shape, outout_shape, embedding_dim, n_features, num_cosines=128, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = outout_shape[0]

        self.embedding_dim = embedding_dim

        self._state_net = build_mlp(n_input, n_features, embedding_dim, 1)

        self._cosine_net = CosineEmbeddingNetwork(num_cosines, embedding_dim)

        self._quantile_net = build_mlp(embedding_dim, 1, n_output, 1)

        # self._quantile_net[-1].bias.data.fill_(0.5)

    def forward(self, state, action, tau):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        state_embeddings = self._state_net(state_action)  # (B x embedding_dim)

        tau_embeddings = self._cosine_net(tau.float())  # (B x N x embedding_dim)

        batch_size = state_embeddings.shape[0]
        N = tau_embeddings.shape[1]

        # Reshape into (batch_size, 1, embedding_dim).
        state_embeddings = state_embeddings.view(
            batch_size, 1, self.embedding_dim)

        # Calculate embeddings of states and taus.
        embeddings = (state_embeddings * tau_embeddings).view(
            batch_size * N, self.embedding_dim)

        quantiles = self._quantile_net(embeddings)
        return quantiles.view(batch_size, N, -1)


class ImplicitQuantileStateConstraint(nn.Module):
    def __init__(self, input_shape, outout_shape, embedding_dim, n_features, num_cosines=128, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = outout_shape[0]

        self.embedding_dim = embedding_dim

        self._state_net = build_mlp(n_input, n_features, embedding_dim, 1)

        self._cosine_net = CosineEmbeddingNetwork(num_cosines, embedding_dim)

        self._quantile_net = build_mlp(embedding_dim, 1, n_output, 1)

        # self._quantile_net[-1].bias.data.fill_(0.5)

    def forward(self, state, tau):
        state_embeddings = self._state_net(torch.atleast_2d(state.float()))  # (B x embedding_dim)

        tau_embeddings = self._cosine_net(tau.float())  # (B x N x embedding_dim)

        batch_size = state_embeddings.shape[0]
        N = tau_embeddings.shape[1]

        # Reshape into (batch_size, 1, embedding_dim).
        state_embeddings = state_embeddings.view(
            batch_size, 1, self.embedding_dim)

        # Calculate embeddings of states and taus.
        embeddings = (state_embeddings * tau_embeddings).view(
            batch_size * N, self.embedding_dim)

        quantiles = self._quantile_net(embeddings)
        return quantiles.view(batch_size, N)


# From https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch/blob/master/fqf_iqn_qrdqn/network.py
class CosineEmbeddingNetwork(nn.Module):

    def __init__(self, num_cosines=64, embedding_dim=128):
        super(CosineEmbeddingNetwork, self).__init__()

        self.net = build_mlp(num_cosines, None, embedding_dim, 0)
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines + 1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
        ).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.embedding_dim)

        return tau_embeddings


class QuantileCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, activation, embedding_size,
                 dropout_ratio=0, layer_norm=False, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features = n_features
        n_features.insert(0, n_input)
        n_features.append(n_output)

        self.base_net = nn.Sequential()
        for i in range(len(n_features[:-2])):
            layer = nn.Linear(n_features[i], n_features[i + 1])
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(activation))
            self.base_net.append(layer)
            if dropout_ratio > 0:
                self.base_net.append(nn.Dropout(dropout_ratio))
            if layer_norm:
                self.base_net.append(nn.LayerNorm(n_features[i + 1]))
            self.base_net.append(ActivationLayer[activation]())

        self.embedding_net = nn.Sequential(
            nn.Linear(embedding_size, n_features[-2]), nn.Sigmoid())  # Sigmoid used in DSAC
        self.register_buffer('embed_vec', torch.arange(0, embedding_size, 1).float())

        self.quantile_net = nn.Sequential()
        self.quantile_net.append(nn.Linear(n_features[-2], n_features[-2]))
        nn.init.xavier_uniform_(self.quantile_net[-1].weight, gain=nn.init.calculate_gain(activation))
        if layer_norm:
            self.quantile_net.append(nn.LayerNorm(n_features[-2]))
        self.quantile_net.append(ActivationLayer[activation]())
        self.quantile_net.append(nn.Linear(n_features[-2], n_features[-1]))
        nn.init.xavier_uniform_(self.quantile_net[-1].weight, gain=nn.init.calculate_gain('linear'))
        self.quantile_net.append(nn.Softplus())

    def forward(self, state, tau):
        # assert tau.dim() == 2 and tau.shape[0] == state.shape[0]
        # state_action = torch.cat((state.float(), action.float()), dim=-1)  # (B, S + A)
        state_action_embedding = self.base_net(state.float())  # (B, F)
        tau_embedding = self.embedding_net(torch.cos(torch.pi * self.embed_vec *
                                                     tau.float().unsqueeze(-1)))  # (B, T, F)

        assert state_action_embedding.shape[-1] == tau_embedding.shape[-1]

        quantiles = self.quantile_net(state_action_embedding.unsqueeze(-2) * tau_embedding)  # (B, T, 1)
        return quantiles.squeeze(-1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    net = MLP((2,), (1,), [256])

    print(net)

    N = 200
    X, Y = np.mgrid[-2:2:complex(0, N), -2:2:complex(0, N)]
    positions = np.vstack([X.ravel(), Y.ravel()]).T

    with torch.no_grad():
        y_mu = net.forward(torch.from_numpy(positions).type(torch.FloatTensor))
    # print(y_mu, y_mu.min(), y_mu.max())
    t = plt.scatter(*positions.T, c=y_mu)
    plt.colorbar(t)
    plt.show()
