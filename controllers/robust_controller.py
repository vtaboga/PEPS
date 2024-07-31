import gin
import jax
import jax.numpy as jnp
import ray

from controllers.abstract_controller import ABCController
from typing import Dict, Union


@gin.configurable
class RobustController(ABCController):

    def __init__(
        self,
        prediction_model,
        seed: int,
        model_path: str,
        planning_timesteps: int,
        temperature_weight: float,
        mininum_setpoint_change: float,
        maximum_setpoint_change: float,
        setpoint_change_step: float,
        n_setpoint_trajectories: int,
        n_random_trajectories: int,
        mean_constants: Dict,
        std_constants: Dict,
        state_indexes: Dict
    ):

        ABCController.__init__(self)
        if type(prediction_model) is type:
            # instantiate the class
            self.prediction_model = prediction_model(
                mean=mean_constants,
                std=std_constants,
                state_indexes=state_indexes,
                model_path=model_path
            )
        else:
            self.prediction_model = prediction_model

        self.setpoint_change_index = state_indexes['setpoint_change']
        self.planning_timesteps = planning_timesteps
        self.mininum_setpoint_change = mininum_setpoint_change
        self.maximum_setpoint_change = maximum_setpoint_change
        self.setpoint_change_step = setpoint_change_step
        self.action_space = jnp.arange(
            self.mininum_setpoint_change,
            self.maximum_setpoint_change + self.setpoint_change_step,
            self.setpoint_change_step
        )
        self.prediction_horizon = self.prediction_model.prediction_horizon
        self.temperature_weight = temperature_weight
        self.norm_temperature_obj = jnp.linalg.norm(
            jnp.ones(self.prediction_horizon) * self.mininum_setpoint_change
        ) ** 2

        self.n_setpoint_trajectories = n_setpoint_trajectories
        self.n_random_trajectories = n_random_trajectories
        key = jax.random.key(seed)
        subkey, self.key = jax.random.split(key)
        self.setpoint_changes_trajectories = jnp.zeros((self.n_setpoint_trajectories, self.prediction_horizon))
        self.state_trajectories = jnp.zeros((self.n_setpoint_trajectories, self.prediction_horizon))
        self.draw_trajectories = True

    def reset_controller(self) -> None:
        self.draw_trajectories = True

    def run(
        self,
        observation: jnp.array,
        weather: jnp.array,
        weather_lags,
        action_lags,
        power_schedule: jnp.array,
        temperature_targets: Union[None, jnp.array],
        time_to_start_dr_event,
        time_to_end_dr_event,
        energyplus_timestep_duration: int,
        logs_file: str,
        dual_variables: jnp.array = None,
        rho: float = None
    ):

        """
        :param observation:
        :param weather:
        :param weather_lags:
        :param action_lags:
        :param power_schedule:
        :param temperature_targets:
        :param time_to_start_dr_event:
        :param time_to_end_dr_event:
        :param energyplus_timestep_duration:
        :param logs_file:
        :param dual_variables:
        :param rho:
        :return:
        """

        if self.draw_trajectories:
            # draw new trajectories
            subkey1, subkey2, self.key = jax.random.split(self.key, 3)
            self.setpoint_changes_trajectories = self.draw_setpoint_change_trajectories(subkey1)
            subkeys = jax.random.split(subkey2, self.n_setpoint_trajectories)
            self.state_trajectories = jax.vmap(
                self.draw_worst_state_trajectories, in_axes=(None, None, 0, None, None, None, 0))(
                    observation,
                    weather,
                    self.setpoint_changes_trajectories,
                    weather_lags,
                    action_lags,
                    energyplus_timestep_duration,
                    subkeys
                )
            self.draw_trajectories = False

        trajectories_score = jax.vmap(self.evaluate_trajectory, in_axes=(0, 0, None, None, None))(
            self.state_trajectories,
            self.setpoint_changes_trajectories,
            dual_variables,
            power_schedule,
            rho
        )
        best_trajectory_idx = jnp.argmin(trajectories_score)

        return self.setpoint_changes_trajectories[best_trajectory_idx], self.state_trajectories[best_trajectory_idx]

    def draw_worst_state_trajectories(
        self,
        current_obs,
        weather_forecast,
        setpoint_changes,
        weather_lags,
        action_lags,
        energyplus_timestep_duration,
        key
    ) -> jnp.array:
        """
        :param current_obs:
        :param weather_forecast:
        :param setpoint_changes:
        :param weather_lags:
        :param action_lags:
        :param key:
        :return:
        """

        subkeys = jax.random.split(key, self.n_setpoint_trajectories)
        trajectories = jax.vmap(self.draw_state_trajectory, in_axes=(None, None, None, None, None, None, 0))(
            current_obs,
            weather_forecast,
            setpoint_changes,
            weather_lags,
            action_lags,
            energyplus_timestep_duration,
            subkeys
        )
        worst_trajectory = self.select_worst_trajectory(trajectories)

        return worst_trajectory

    def draw_state_trajectory(
        self,
        current_obs,
        weather_forecast,
        setpoint_changes,
        weather_lags,
        action_lags,
        energyplus_timestep_duration,
        key
    ) -> jnp.array:
        """
        :param weather_forecast:
        :param setpoint_changes:
        :param weather_lags:
        :param action_lags:
        :param key:
        :return:
        """

        inputs = self.process_input_data(
            observations=current_obs,
            weather_forecast=weather_forecast,
            actions_forecast=setpoint_changes,
            weather_lags=weather_lags,
            actions_lags=action_lags
        )

        predictions = self.prediction_model.sample_predictions(
            inputs=inputs['inputs'],
            actions=inputs['actions'],
            energyplus_timestep_duration=energyplus_timestep_duration,
            prediction_horizon=self.prediction_horizon,
            key=key,
        )
        # keep only power prediction
        power_prediction = predictions[:, 0]
        power_prediction = jnp.where(power_prediction < 0, 0, power_prediction)  # no negative powers

        return power_prediction

    def select_worst_trajectory(self, state_trajectories: jnp.array) -> jnp.array:
        """
        :param state_trajectories: power trajectories drawn from the same setpoint trajectory. shape (n, horizon)
        :return: state trajectory with the maximum power consumption
        """

        idx = jnp.argmax(jnp.max(state_trajectories, axis=1))
        worst_trajectory = state_trajectories[idx]
        return worst_trajectory

    def draw_setpoint_change_trajectories(self, key) -> jnp.array:
        """
        :param key: Jax random key generator
        :return: Create trajectories of setpoint changes. Return an array of shape (n_action_trajectries, horizon)
        """

        n_actions = (self.prediction_horizon // self.planning_timesteps)
        all_trajectories = self.action_space.shape[0] ** n_actions
        if all_trajectories <= self.n_setpoint_trajectories:
            indices = jnp.arange(self.action_space.shape[0])
            grids = jnp.meshgrid(*([indices] * n_actions))
            combinations = jnp.stack(grids, axis=-1).reshape(-1, n_actions)
            trajectories = self.action_space[combinations]
            trajectories = jnp.repeat(trajectories, self.planning_timesteps, axis=1)
            self.n_setpoint_trajectories = trajectories.shape[0]
        else:
            # draw the trajectories randomly
            # initialize with predifined trajectories
            no_change = jnp.zeros(self.prediction_horizon)
            min_setpoint = jnp.ones(self.prediction_horizon) * self.mininum_setpoint_change
            mid_setpoint = jnp.ones(self.prediction_horizon) * int(
                (self.maximum_setpoint_change - self.mininum_setpoint_change) / 2
            )
            predefined_trajectories = [no_change, min_setpoint, mid_setpoint]
            # draw random trajectories
            n = self.n_setpoint_trajectories - len(predefined_trajectories)
            subkeys = jax.random.split(key, n)
            trajectories = jax.vmap(self.draw_setpoint_change_trajectory)(subkeys)
            trajectories = jnp.vstack(predefined_trajectories + [trajectories])

        return trajectories

    def draw_setpoint_change_trajectory(self, key: jnp.array) -> jnp.array:
        """
        :param key: Jax random key generator
        :return: array of shape (horizon)
        Generate a random sequence of setpoint changes.
        """

        n = self.prediction_horizon // self.planning_timesteps
        subkeys = jax.random.split(key, num=n)
        actions = jax.vmap(lambda subkey: jax.random.choice(subkey, self.action_space, shape=(1,)))(subkeys)
        actions_repeated = jnp.repeat(actions, self.planning_timesteps, axis=0)
        setpoint_changes = jnp.reshape(actions_repeated, (-1,))

        return setpoint_changes

    def evaluate_trajectory(self, state_trajectory, setpoint_changes, dual_variables, power_schedule, rho) -> float:

        error_temperature = jnp.linalg.norm(setpoint_changes) ** 2
        error_power = jnp.dot(dual_variables, power_schedule - state_trajectory)
        error_power += rho * jnp.linalg.norm(power_schedule - state_trajectory) ** 2
        error = error_temperature / self.norm_temperature_obj + error_power
        return error

    def make_predictions(
        self,
        current_obs: jnp.array,
        setpoint_changes: jnp.array,
        weather_forecast: jnp.array,
        weather_lags: Union[None, jnp.array],
        action_lags: Union[None, jnp.array],
        energyplus_timestep_duration: int,
        prediction_horizon: int,
    ) -> Dict:
        """
        :param current_obs:
        :param actions_forecast:
        :param weather_forecast:
        :param energyplus_timestep_duration:
        :param horizon:
        :return:
        """

        inputs = self.process_input_data(
            observations=current_obs,
            weather_forecast=weather_forecast,
            actions_forecast=setpoint_changes,
            weather_lags=weather_lags,
            actions_lags=action_lags
        )

        subkey, self.key = jax.random.split(self.key)
        predictions = self.prediction_model.make_predictions(
            inputs=inputs['inputs'],
            actions=inputs['actions'],
            energyplus_timestep_duration=energyplus_timestep_duration,
            prediction_horizon=prediction_horizon,
            key=subkey
        )
        # unpack results
        power_prediction = predictions[:, 0]
        power_prediction = jnp.where(power_prediction < 0, 0, power_prediction)  # no negative powers
        temperature_prediction = predictions[:, 1]
        predictions = {'power': power_prediction, 'temperature': temperature_prediction}

        return predictions


@ray.remote
class RayRobustController(RobustController):

    def __init__(
        self,
        prediction_model,
        seed,
        model_path: str,
        planning_timesteps: int,
        temperature_weight: float,
        mininum_setpoint_change: float,
        maximum_setpoint_change: float,
        setpoint_change_step: float,
        n_setpoint_trajectories: int,
        n_random_trajectories: int,
        mean_constants: Dict,
        std_constants: Dict,
        state_indexes: Dict
    ):
        jax.config.update("jax_enable_x64", True)
        RobustController.__init__(
            self,
            prediction_model=prediction_model,
            seed=seed,
            model_path=model_path,
            planning_timesteps=planning_timesteps,
            temperature_weight=temperature_weight,
            mininum_setpoint_change=mininum_setpoint_change,
            maximum_setpoint_change=maximum_setpoint_change,
            setpoint_change_step=setpoint_change_step,
            mean_constants=mean_constants,
            std_constants=std_constants,
            n_setpoint_trajectories=n_setpoint_trajectories,
            n_random_trajectories=n_random_trajectories,
            state_indexes=state_indexes
        )

    def get_minimum_setpoint_change(self) -> float:
        return self.mininum_setpoint_change

    def get_maximum_setpoint_change(self) -> float:
        return self.maximum_setpoint_change
