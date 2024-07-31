from abc import ABC
from abc import abstractmethod


class ABCCoordinator(ABC):

    @abstractmethod
    def run(self, lc_powers):
        """find aggregator power constraints (vk + bar(uk)) to send to the LCs"""
        pass

    @abstractmethod
    def update_dual_variables(self):
        pass

    @abstractmethod
    def send_dual_variables(self):
        pass

    @abstractmethod
    def send_power_targets(self):
        pass

    @abstractmethod
    def check_convergence(self):
        pass





