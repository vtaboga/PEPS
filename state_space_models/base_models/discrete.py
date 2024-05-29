from state_space_models.base_models.haiku_networks import *
from state_space_models.base_models.ode import CalorificCapacity


class PredictionModel:

    def __init__(
        self,
        decoder_depth,
        encoder_dim,
        decoder_dim,
        hidden_dim,
        output_dim
    ):
        self.encoder = RNNEncoder(encoded_obs_size=hidden_dim, rnn_hidden_size=encoder_dim)
        self.gru = hk.GRU(hidden_size=hidden_dim)
        self.decoder = MLP(
            output_dim=output_dim,
            depth=decoder_depth,
            hidden_dim=decoder_dim
        )

    def __call__(self, inputs, actions):
        latent_state = self.encoder(inputs)

        def scan_fn(carry, input):
            _, new_state = self.gru(input, carry)
            return new_state, new_state

        _, trajectory = hk.scan(scan_fn, latent_state, actions)
        decoded_trajectory = hk.vmap(self.decoder, split_rng=False)(trajectory)

        return decoded_trajectory


class PhysicsInformedPredictionModel:
    def __init__(
        self,
        decoder_depth: int,
        encoder_dim: int,
        decoder_dim: int,
        hidden_dim: int,
    ):
        self.hidden_dim = hidden_dim
        self.encoder = LatentRNNEncoder(output_dim=hidden_dim, rnn_hidden_size=encoder_dim)
        self.gru = hk.GRU(hidden_size=hidden_dim)
        self.efficiency = lambda x: x @ hk.get_parameter(
            "efficiency_matrix",
            shape=(hidden_dim // 2, hidden_dim // 2),
            init=hk.initializers.RandomNormal()
        )
        self.Cp = CalorificCapacity(final_activation='tanh')
        self.decoder = LatentDecoder(latent_space_dim=hidden_dim, depth=decoder_depth, hidden_dim=decoder_dim)

    def __call__(self, state, actions):
        latent_state = self.encoder(state)

        def scan_fn(carry, input):
            _, new_powers = self.gru(input, carry)
            hvac_power, non_hvac_power = new_powers[:self.hidden_dim//2], new_powers[self.hidden_dim//2:]
            hvac_heat = self.efficiency(hvac_power)
            temperature = self.Cp(non_hvac_power, hvac_heat)
            new_state = jnp.concatenate([hvac_power, temperature])

            return new_state, new_state

        _, trajectory = hk.scan(scan_fn, latent_state, actions)

        decoded_trajectory = hk.vmap(self.decoder, split_rng=False)(trajectory)

        return decoded_trajectory
