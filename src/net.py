"""
Module containing neuron, layer and network classes
"""

from __future__ import annotations
from src.helpers import random_adjustment


class DNFNet:
    """
    Network class representing a DNF
    """
    def __init__(self, input_length: int = 10, num_monomers: int = 5, learning_rate: float = 0.1):
        """
        Initializes a network with random weights and biases.
        Args:
            input_length (int, optional): Number of inputs to the network. Defaults to 10.
            num_monomers (int, optional): Number of monomers in the network. Defaults to 5.
            learning_rate (float, optional): Learning rate for the network. Defaults to 0.1.
        """
        self.input_length = input_length
        self.num_monomers = num_monomers
        self.learning_rate = learning_rate

        self.monomer_weights = [[random_adjustment() for _ in range(self.input_length)] for _ in range(self.num_monomers)]
        self.monomer_biases = [random_adjustment() for _ in range(self.num_monomers)]

        self.output_weights = [random_adjustment() for _ in range(self.num_monomers)]
        self.output_bias = random_adjustment()


    def inference(self, inputs: list[int]) -> tuple[int, list[int]]:
        """
        Returns the output of the neuron given an input.
        Args:
            inputs (list[int]): List of inputs to the network.
        
        Returns:
            tuple[int, list[int]]: Output of the Network and list of monomer activations.
        """
        assert len(inputs) == self.input_length
        output_activation = 0
        monomer_activations = []

        # Forward pass (iterate over monomers)
        for i in range(self.num_monomers):
            # Calculate the activation of the monomer
            cumulative_connections = 0.0

            # Iterate over inputs and calculate the connections element-wise
            for j in range(self.input_length):
                cumulative_connections += self.monomer_weights[i][j] * inputs[j]

            # Calculate the activation of the monomer
            monomer_activation = self.activation(cumulative_connections - self.monomer_biases[i])

            # Store the monomer activation for backpropagation
            monomer_activations.append(monomer_activation)

            # Add to element-wise sum of the monomer activation in the output
            output_activation += self.output_weights[i] * monomer_activation

        output_activation = self.activation(output_activation - self.output_bias)

        return output_activation, monomer_activations


    def backpropagation(self, inputs: list[int], target: int, result: int, monomer_activations: list[int]) -> None:
        """
        Backpropagates the error through the network.
        
        Args:
            inputs (list[int]): Inputs to the network.
            target (int): Target output.
            result (int): Ouput of the network for the given input.
            monomer_activations (list[int]): Activations of the monomers.
        """
        assert len(inputs) == self.input_length
        # Iterate over monomer weights
        for i in range(self.num_monomers):
            # Iterate over input weights
            for j in range(self.input_length):
                # Update the weight
                self.monomer_weights[i][j] += self.learning_rate * self.output_weights[i] * (target - result) * inputs[j]

            # Update the bias
            self.monomer_biases[i] -= self.learning_rate * self.output_weights[i] * (target - result)

        # Update the output weights
        for i in range(self.num_monomers):
            self.output_weights[i] += self.learning_rate * (target - result) * monomer_activations[i]

        # Update the output bias
        self.output_bias -= self.learning_rate * (target - result)


    def train(self, inputs: list[int], target: int) -> tuple[int, list[int]]:
        """
        Trains the network with a given input and target output.
        
        Args:
            inputs (list[int]): List of inputs to the network.
            target (int): Expected output of the network.
        
        Returns:
            tuple[int, list[int]]: The output of the network and the activations of the monomers.
        """
        activation, monomer_activations = self.inference(inputs)

        # Update the weights and bias of the monomer
        self.backpropagation(inputs, target, activation, monomer_activations)

        return activation, monomer_activations


    def activation(self, num: float | int):
        """
        Network's activation function. Returns the sign of a number.
        """
        return 1 if num >= 0 else -1


    def __str__(self) -> str:
        """
        Returns a string representation of the network.
        Returns:
            str: String representation of the network.
        """
        string = f"Network with {self.input_length} inputs and {self.num_monomers} monomers\n\n"

        for i in range(self.num_monomers):
            string += f"Monomer {i}: Weights = {self.monomer_weights[i]} Bias = {self.monomer_biases[i]}\n\n"

        string += f"Output weights: {self.output_weights}\n"
        string += f"Output bias: {self.output_bias}\n"
        return string


    def __call__(self, inputs: list[int], target: int = 0, train: bool = False) -> tuple[int, list[int]]:
        """
        Calls the network with a given input and target output. Performs a training iteration with parameter adjustments if train is set to True, otherwise performs inference.

        Args:
            inputs (list[int]): list of integers in {-1, 1} representing the input, i.e. the truth assignment of the variables
            target (int, optional): Expected output of the network for the given input. Should be in {-1, 1} if specified. Defaults to 0.
            train (bool, optional): _description_. Defaults to False.

        Returns:
            tuple[int, list[int]]: Tuple of the output of the network and the output of the monomers
        """
        if train:
            return self.train(inputs, target)
        return self.inference(inputs)


    def __eq__(self, value: object) -> bool:
        """
        Checks if two networks are equal in parameters, except for the learning rate metaparameter. Returns False if value is not a DNFNet.

        Args:
            value (DNFNet): DNFNet that should be compared to self.

        Returns:
            bool: Boolean indicating whether the two networks are equal (True) or not (False)
        """
        if not isinstance(value, DNFNet):
            return False
        return self.monomer_weights == value.monomer_weights and self.monomer_biases == value.monomer_biases and self.output_weights == value.output_weights and self.output_bias == value.output_bias


    @property
    def shape(self) -> tuple[int, int]:
        """
        Returns the shape of the network, i.e. the number of inputs and the number of monomers.
        Returns:
            tuple[int, int]: Tuple of the number of inputs and the number of monomers.
        """
        return (self.input_length, self.num_monomers)
