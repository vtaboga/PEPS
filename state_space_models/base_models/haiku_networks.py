import haiku as hk
import jax.numpy as jnp
import jax


class MLP(hk.Module):
    def __init__(self, output_dim: int, depth: int, hidden_dim: int, act: str = 'relu', is_vector_field: bool = False):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.is_vector_field = is_vector_field
        if act != 'relu':
            raise ValueError(f'activation type {act} is not implemented yet')

    def __call__(self, x):
        for _ in range(self.depth):
            x = hk.Linear(self.hidden_dim)(x)
            x = jax.nn.relu(x)
        x = hk.Linear(self.output_dim)(x)
        if self.is_vector_field:
            x = jax.nn.tanh(x)
        return x


class RNNEncoder(hk.Module):

    def __init__(self, encoded_obs_size, rnn_hidden_size):
        super().__init__()
        self.encoded_obs_size = encoded_obs_size
        self.rnn_hidden_size = rnn_hidden_size
        self.gru_cell = hk.GRU(hidden_size=rnn_hidden_size)
        self.linear = hk.Linear(output_size=encoded_obs_size)

    def __call__(self, inputs):

        hidden = jnp.zeros(self.rnn_hidden_size)

        def f(carry, inp):
            return self.gru_cell(inp, carry)

        final_hidden, _ = hk.scan(f, hidden, inputs)
        out = self.linear(final_hidden)

        return out


class LatentEncoder(hk.Module):

    def __init__(self, output_dim, depth, hidden_dim, name=None):
        super().__init__(name=name)
        self.power_encoder = MLP(
            output_dim=output_dim//2,
            depth=depth,
            hidden_dim=hidden_dim
        )
        self.temperature_encoder = MLP(
            output_dim=output_dim//2,
            depth=depth,
            hidden_dim=hidden_dim
        )

    def __call__(self, power_lags, temperature_lags):
        latent_state = jnp.concatenate(
            [
                self.power_encoder(power_lags),
                self.temperature_encoder(temperature_lags)
            ]
        )

        return latent_state


class LatentDecoder(hk.Module):

    def __init__(self, latent_space_dim, depth, hidden_dim, name=None):
        super().__init__(name=name)
        self.latent_space_dim = latent_space_dim
        self.power_decoder = MLP(
            output_dim=1,
            depth=depth,
            hidden_dim=hidden_dim
        )
        self.temperature_decoder = MLP(
            output_dim=1,
            depth=depth,
            hidden_dim=hidden_dim
        )

    def __call__(self, latent_state):
        state = jnp.array(
            [
                self.power_decoder(latent_state[:self.latent_space_dim // 2]),
                self.temperature_decoder(latent_state[-self.latent_space_dim // 2:])
            ]
        )
        return jnp.squeeze(state)


class LatentRNNEncoder(hk.Module):

    def __init__(self, output_dim, rnn_hidden_size, name=None):
        super().__init__(name=name)
        self.power_encoder = RNNEncoder(
            encoded_obs_size=output_dim//2,
            rnn_hidden_size=rnn_hidden_size,
        )
        self.temperature_encoder = RNNEncoder(
            encoded_obs_size=output_dim//2,
            rnn_hidden_size=rnn_hidden_size
        )

    def __call__(self, inputs):
        latent_state = jnp.concatenate(
            [
                self.power_encoder(inputs),
                self.temperature_encoder(inputs)
            ]
        )

        return latent_state
