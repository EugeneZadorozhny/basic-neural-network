"""
* This module contains the class that allows us to record our program's
runtime metrics
"""


class RuntimeMetrics:
    def __init__(self, size: int, time_ns: float):
        """
        Holds data to determine Big-O runtime metrics of our program.
        :param size: Size of the input  passed to the network
        :param time_ns: Duration of the run
        """
        self.size = size
        self.time = time_ns

    def get_runtime(self) -> float:
        """
        Returns the time it took to complete the run
        :return: The time measured in nanoseconds
        """

        return self.time

    def get_size(self) -> int:
        """
        Returns the size of the problem, in our case the size of the input
        to the network
        :return: The size of the input to the network
        """
        return self.size
