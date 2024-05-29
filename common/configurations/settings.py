import gin


@gin.configurable
class Settings:
    def __init__(self, jax_enable_x64, simulation_seed):
        self.jax_enable_x64 = jax_enable_x64
        self.simulation_seed = simulation_seed
