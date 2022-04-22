import torch
from torch.autograd import Function
import numpy as np

import qiskit
from quantum_circuit import QuanConvCircuit

use_cuda = torch.cuda.is_available()


class QuanConvFunction(Function):
    """QuanconvFunction
        This class defines quantum convolution function
    Args:
        Function (Class): Records operation history and defines formulas for differentiating ops for PyTorch function
    """
    @staticmethod
    def forward(ctx, inputs, in_channels, out_channels, kernel_size, quantum_circuits, shift):
        """Forward pass computation function
        Args:
            ctx: ctx is a context object that can be used to stash information for backward computation. You can cache arbitrary objects for use in the backward pass using the ctx.save_for_backward method.
            inputs (Tensor): a Tensor containing the input
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int): Size of the convolving kernel
            quantum_circuits (_type_): _description_
            shift (float): The parameter-shift rule is an approach to measuring gradients of quantum circuits with respect to their parameters, which does not require ancilla qubits or controlled operations
        Returns: a Tensor containing the output
        Example:
            input shape : (-1, 1, 28, 28)
            output shape : (-1, 6, 24, 24)
        """
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.kernel_size = kernel_size
        ctx.quantum_circuits = quantum_circuits
        ctx.shift = shift

        _, _, len_x, len_y = inputs.size()
        len_x = len_x - kernel_size + 1
        len_y = len_y - kernel_size + 1

        if use_cuda:
            outputs = torch.zeros(
                (len(inputs), len(quantum_circuits), len_x, len_y)).cuda()
        else:
            outputs = torch.zeros(
                (len(inputs), len(quantum_circuits), len_x, len_y))

        for i in range(len(inputs)):
            input = inputs[i]
            for c in range(len(quantum_circuits)):
                circuit = quantum_circuits[c]
                for h in range(len_y):
                    for w in range(len_x):
                        data = input[0, h:h+kernel_size, w:w+kernel_size]
                        outputs[i, c, h, w] = circuit.run(data)

        ctx.save_for_backward(inputs, outputs)
        return outputs
        '''
        features = []
        for input in inputs:
            feature = []
            for circuit in quantum_circuits:
                xys = []
                for x in range(len_x):
                    ys = []
                    for y in range(len_y):
                        data = input[0, x:x+kernel_size, y:y+kernel_size]
                        ys.append(circuit.run(data))
                    xys.append(ys)
                feature.append(xys)
            features.append(feature)
        result = torch.tensor(features)

        ctx.save_for_backward(inputs, result)
        return result
        '''
    @staticmethod
    def backward(ctx, grad_outputs):
        """Backward pass computation function

        Args:
            ctx: ctx is a context object that can be used to stash information for backward computation
            grad_outputs (Tensor): a Tensor containing the gradient outputs
        """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            if use_cuda:
                gradient = torch.tensor([expectation_right]).cuda() - \
                    torch.tensor([expectation_left]).cuda()
            else:
                gradient = torch.tensor([expectation_right]) - \
                    torch.tensor([expectation_left])
            gradients.append(gradient)

        if use_cuda:
            gradients = torch.tensor([gradients]).cuda()
            gradients = torch.transpose(gradients, 0, 1)
        else:
            gradients = torch.tensor([gradients])
            gradients = torch.transpose(gradients, 0, 1)

        return gradients.float() * grad_outputs.float(), None, None


class QuanConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, back_end=qiskit.Aer.get_backend('qasm_simulator'), shots=100, shift=np.pi/2):
        """_summary_

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int): Size of the convolving kernel
            back_end (qiskit.Aer.backend, optional): _description_. Defaults to qiskit.Aer.get_backend('qasm_simulator').
            shots (int, optional): How many shots we wish to use in our quantum circuit. Defaults to 100.
            shift (float, optional): _description_. Defaults to np.pi/2. Read "Gradients of parameterized quantum gates using the parameter-shift rule and gate decomposition" for more information about shift parameter https://arxiv.org/pdf/1905.13311.pdf
        back_end support list:
            aer_simulator
            aer_simulator_statevector
            aer_simulator_density_matrix
            aer_simulator_stabilizer
            aer_simulator_matrix_product_state
            aer_simulator_extended_stabilizer
            aer_simulator_unitary
            aer_simulator_superop
            qasm_simulator
            statevector_simulator
            unitary_simulator
            pulse_simulator
        """
        super(QuanConv, self).__init__()
        self.quantum_circuits = [QuanConvCircuit(
            kernel_size=kernel_size, back_end=back_end, shots=shots, threshold=127)]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.shift = shift

    def forward(self, inputs):
        """Define forward pass

        Args:
            nputs (Tensor): a Tensor containing the input
        """
        return QuanConvFunction.apply(inputs, self.in_channels, self.out_channels, self.kernel_size, self.quantum_circuit, self.shift)
