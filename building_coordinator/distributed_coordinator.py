import os.path

import gin
import numpy as np
import pickle

from scipy.optimize import minimize, Bounds
from building_coordinator.abstract_coordinator import ABCCoordinator


@gin.configurable
class DistributedCoordinator(ABCCoordinator):

    def __init__(
        self,
        epsilon: float,
        rho: float,
        number_of_zones: int,
        horizon: int,
        slack_maximum_power: float,
        maximum_admm_iterations: int,
        planning_timestep,
        weight_objective: float,
    ):

        """
        :param epsilon:
        :param rho:
        :param number_of_zones: number of zones being controlled in the building
        :param horizon: number of timesteps in the prediction horizon
        :param planning_timesteps:
        :param weight_objective:
        """

        self.iteration_number = 0  # admm iteration number
        self.epsilon = epsilon
        self.rho = rho
        self.n_zones = number_of_zones
        self.horizon = horizon
        self.slack_maximum_power = slack_maximum_power
        self.maximum_admm_iterations = maximum_admm_iterations
        self.lc_powers = np.zeros((self.horizon, self.n_zones))
        self.power_targets = np.zeros((self.horizon, self.n_zones))
        self.dual_variables = np.ones((self.horizon, self.n_zones))
        self.has_converge = False
        self.total_power = None
        self.weight_objective = weight_objective
        self.planning_timesteps = planning_timestep

        self.memory = None
        self._total_power_matrix = None
        self._init_total_power_matrix()
        self.reset()

    def run(self, lc_powers: np.array) -> (np.array, bool):
        """Solve the aggregator optimization problem given the current LC powers
        returns the new aggregator constraints for the lc problem and a boolean for convergence"""

        lc_powers = np.transpose(lc_powers)  # array of shape (horizon, n zones)
        self.lc_powers = lc_powers
        bnds = Bounds(np.zeros(self.n_zones * self.horizon),
                      np.inf)
        res = minimize(
            fun=self._objective,
            x0=self.power_targets.flatten(),
            bounds=bnds,
            options={'disp': False}
        )
        self.power_targets = self._unflatten(res.x)
        self.update_dual_variables()
        self.check_convergence()
        self.update_memory()
        self.iteration_number += 1

        return self.has_converge

    def update_dual_variables(self) -> None:
        self.dual_variables = self.dual_variables + self.rho * (self.power_targets - self.lc_powers)

    def check_convergence(self) -> None:
        """verify if ADMM has converge using the norm of the difference of two consecutive slack variables"""

        if self.iteration_number > 0:
            old_dual_variables = self.memory['dual_variables'][-1]
            eps = np.linalg.norm(old_dual_variables - self.dual_variables)
            print(f"check dual convergence: eps = {eps}")
            self.update_slacks_memory(eps)
            if eps <= self.epsilon:
                self.has_converge = True
        else:
            pass

    def _objective(self, power_targets: np.array):
        """ objective function to minimize
            agg_powers : shape (n_zones * n_horizon)"""
        lc_powers = self.lc_powers.flatten()
        power_objective_agg = np.linalg.norm(self._total_power_matrix @ power_targets - self.total_power)**2
        power_objective = np.dot(self.dual_variables.flatten(), power_targets - lc_powers)
        unflat_coordinator_powers = self._unflatten(power_targets)
        diff = unflat_coordinator_powers - self.lc_powers
        diff_penalty = self.rho * np.linalg.norm(diff, axis=0) ** 2
        power_objective += np.sum(diff_penalty)
        return power_objective + self.weight_objective * power_objective_agg

    def _unflatten(self, arr) -> np.array:
        """arr : array to unflatten
        take an array of shape (horizon * n_zones). return an array of shape (horizon, n zones)
        each row is a zone for the entire horizon """

        unflatten_arr = np.zeros((self.horizon, self.n_zones))
        for t in range(self.horizon):
            unflatten_arr[t, :] = arr[t*self.n_zones:(t+1)*self.n_zones]

        return unflatten_arr

    def _init_total_power_matrix(self) -> None:
        # Create an identity matrix of size (horizon, horizon)
        identity_matrix = np.eye(self.horizon)
        # Repeat each row n_zones times along a new axis, creating a block of ones
        expanded_identity = np.repeat(identity_matrix[:, :, None], self.n_zones, axis=2)
        # Reshape the matrix to the final form (horizon, n_zones * horizon)
        self._total_power_matrix = np.reshape(expanded_identity, (self.horizon, self.n_zones * self.horizon))

    def reset(self) -> None:
        self.has_converge = False
        self.iteration_number = 0
        self.dual_variables = self.dual_variables = np.ones((self.horizon, self.n_zones))
        self.power_targets = np.zeros((self.horizon, self.n_zones))
        self.memory = {'dual_variables': [],
                       'power_targets': [],
                       'lc_powers': [],
                       'slacks': []}

    def update_slacks_memory(self, eps: float) -> None:
        self.memory['slacks'].append(eps)

    def update_memory(self) -> None:
        self.memory['dual_variables'].append(self.dual_variables)
        self.memory['power_targets'].append(self.power_targets)
        self.memory['lc_powers'].append(self.lc_powers)

    def save_memory(self, path: str, iteration: int) -> None:
        file_path = path + f'/admm_logs/sim_iteration_{iteration}.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(self.memory, file)

    def update_total_power(self, power_constraints: np.array):
        self.total_power = power_constraints

    def send_total_power(self):
        return self.total_power

    def send_power_targets(self) -> np.array:
        return self.power_targets

    def send_dual_variables(self) -> np.array:
        return self.dual_variables

    def run_aggregator(self, lc_powers: list) -> (np.array, bool):
        """lc_powers : list of array containing the lc powers for the horizon
        creates a row vector of dimension n_zones * horizon and run the aggregator update

        return the aggregator power constraints to pass to the lc
               a boolean for convergence of the ADMM process"""

        dual_variables, has_converge = self.run(lc_powers)
        return dual_variables, has_converge

    def send_variables_norm(self) -> (list, list, list):
        """Compute the norm of different variables to monitor the process"""

        len_memory = len(self.memory['dual_variables'])
        dual_norm = [np.linalg.norm(self.memory['dual_variables'][i]) for i in range(len_memory)]
        agg_norm = [np.linalg.norm(self.memory['power_targets'][i]) for i in range(len_memory)]
        lc_norm = [np.linalg.norm(self.memory['lc_powers'][i]) for i in range(len_memory)]
        slacks = self.memory['slacks']

        return dual_norm, agg_norm, lc_norm, slacks



