import torch
import qiskit
from qiskit import transpile, assemble
import numpy as np
from qiskit.circuit.random import random_circuit


class QuanConvCircuit:
    """
        This class provides a simple inferface for interaction with quantum convolution circuit.
        It defines filter circuit for quantum convolution layers
    """

    def __init__(self, kernel_size, back_end, shots, threshold) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(
            'theta{}'.format(i)) for i in range(self.n_qubits)]

        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)

        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        # ---------------------------

        self.back_end = back_end
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        # Reshape input data: [1, kernel_size, kernel_size] -> [1, self.n_qubits]
        data = torch.reshape(data, (1, self.n_qubits))

        # Encoding data to parameters
        thetas = []
        for item in data:
            theta = []
            for value in item:
                # val > self.threshold  : |1> - rx(pi)
                if value > self.threshold:
                    theta.append(np.pi)
                else:  # val <= self.threshold : |0> - rx(0)
                    theta.append(0)
            thetas.append(theta)

        param_dict = dict()
        for theta in thetas:
            for idx in range(self.n_qubits):
                param_dict[self.theta[idx]] = theta[idx]
        param_binds = [param_dict]

        # Execute random quantum circuit
        t_qc = transpile(self._circuit, self.back_end)
        qobj = assemble(t_qc, shots=self.shots, parameter_binds=param_binds)
        job = self.back_end.run(qobj)
        result = job.result().get_counts()

        # Decoding the result
        counts = 0
        for key, val in result.items():
            cnt = sum([int(char) for char in key])
            counts += cnt * val

        # Compute probabilities for each state
        probabilities = counts / (self.shots * self.n_qubits)

        return probabilities
