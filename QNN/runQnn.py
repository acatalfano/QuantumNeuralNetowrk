import math
import numpy as np
from random import random

import torch
from torch.autograd import Function
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms

import qsharp
from My.QNN import (RunQNN)

###########################
##### HYPER-PARAMETERS ####
###########################

# number of samples to take when approximating probabilities
sample_count = 10

# number of times to repeat initial input vector
# for preprocessing
repeat_count = 1

# number of convolutional layers
num_layers = 1

# spacing of the control-to-target qubits
control_gap = 1


GateParams = tuple[float, float, float]
LayerParams = tuple[list[GateParams], list[GateParams], int]


class Circuit:
    def __init__(self, repeat_times: int) -> None:
        self.__repeat_times = repeat_times
        self.__num_qubits = None
        self.__layer_params = None
        self.clear_params()

    def measure_expectation(self, feature_vec: list[float], model_params: list[LayerParams]) -> float:
        # preprocess feature_vec
        processed_feature_vec = self.__preprocess(feature_vec)

        # if num_qubits is None: set it via feature_vec
        if self.__num_qubits is None:
            self.__num_qubits = int(math.log2(len(initial_layer)))

        # if layer_params is None: set
        if self.__layer_params is None:
            self.__initialize_params(self.__num_qubits)

        return sum([
            RunQNN.simulate(featureVector=feature_vec, allLayersParams=model_params)
                for i in range(sample_count)
        ]) / sample_count

    def clear_params(self) -> None:
        self.params = None

    def __initialize_params(self, num_qubits: int) -> list[LayerParams]:
        return [
            self.__initialize_single_layer_params(num_qubits)
                for _ in range(num_layers)
        ]

    def __next_pow_of_2(self, n: int) -> int:
        return 2 ** (int(math.log2(n)) + 1)

    def __padding_size(self, n: int) -> int:
        return self.__next_pow_of_2(n) - n

    def __repeat_flat(self, feature_vec: list[float]) -> np.ndarray:
        feature_size = len(feature_vec)
        return np.repeat(
            np.array(feature_vec)
            .reshape(1, feature_size),
            self.__repeat_times,
            axis=0
        ).reshape(feature_size * self.__repeat_times)

    def __preprocess(self, input_vec: list[float]) -> np.ndarray:
        repeated_input = self.__repeat_flat(input_vec)
        features = np.array(
            list(repeated_input) + \
            [
                random() for _ in range(
                    self.__padding_size(
                        len(repeated_input)
                    )
                )
            ]
        )
        norm = np.linalg.norm(features)
        return features / norm

    def __initialize_single_layer_params(self, num_qubits: int) -> LayerParams:
        theta = [(random(), random(), random()) for q in range(num_qubits)]
        controlled_theta = [(random(), random(), random()) for q in range(num_qubits)]
        return (theta, controlled_theta, control_gap)

# def run_iteration(feature_vec: list[float], model_params: list[LayerParams]):


#     x = RunQNN.simulate(featureVector=feature_vec, allLayersParams=model_params)
#     print(x)




class NeuralNet(nn.Module):
    def __init__(self, shift):
        super(NeuralNet, self).__init__()
        self.circuit = Circuit()
        self.shift = shift

    def forward(self, input):
        pass

class NeuralNetFunction(Function):
    @staticmethod
    def forward(ctx, input_vec: list[float], params: list[LayerParams], circuit: Circuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit

        expectation = ctx.circuit.measure_expectation(input_vec, params)
        result = torch.tensor([expectation])
        ctx.save_for_backward(input_vec, params, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        input_vec, params, expectation = ctx.saved_tensors
        model_params = np.array(params.tolist())

        shift_right = ctx.__build_shifted_params(False)
        shift_left  = ctx.__build_shifted_params(True)

        # gradients = [x for ]
        gradients = []
        for layer in model_params:
            # theta list
            for thetas in layer[0]:
                pass
            # controlled theta list
            for ctl_thetas in layer[1]:
                pass
            # control gap
            for gap in layer[2]:
                pass

        # TODO: finish

    def __build_shifted_params(ctx, shift_left: bool):
        _, model_layers = ctx.saved_tensors
        return [
            [
                (
                    # (-1) ** shift_left to facilitate -(1-vector) if shift_left and +(1-vector) if shift_right
                    np.array(thetas) +
                        np.ones(np.shape(thetas)) * ((-1) ** shift_left), # theta
                    np.array(controlled_thetas) +
                        np.ones(np.shape(controlled_thetas)) * ((-1) ** shift_left), # controlled_theta
                    control_gap # control_gap
                )
            ] for (thetas, controlled_thetas, control_gap) in model_layers
        ]

    def __compute_gradient_term(self, ctx, input_vec, shift_left_term, shift_right_term):
        # TODO: w/e you do to get the result
        expectation_left = ctx.circuit.measure_expectation(input_vec, shift_left_term)
        expectation_right = ctx.circuit.measure_expectation(input_vec, shift_right_term)
        return torch.tensor([expectation_right]) - torch.tensor([expectation_left])



feature_vec = [1.,2.,3.]

circuit = Circuit(repeat_count)

circuit.measure_expectation(feature_vec)
