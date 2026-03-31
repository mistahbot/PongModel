import numpy as np
import random
import json

class NeuralNetwork:
    def __init__(self, *layers):
        if len(layers) < 2:
            raise ValueError("Need at least input and output layers")

        self.nr_layers = len(layers)

        self.W:list[np.ndarray] = []
        """
        self.W[i] contains the matrix with the weight from layer i - 1 to layer i \n
        only from the index 1 onwards, do the matrices matter \n
        the one at index 0 is there just for padding purposes \n
        important: the input layer is the 0-th layer
        """
        self.W.append(np.zeros((1, 1)))
        for i in range (1, len(layers)):
            self.W.append(np.random.normal(0, np.sqrt(2 / layers[i - 1]), (layers[i], layers[i - 1])))

        self.sum_dW:list[np.ndarray] = []
        """
        self.sum_dW[i] contains the sum of the partial derivatives matrices (over all the backpropagations) for the weight matrix from layer i - 1 to layer i \n
        only from the index 1 onwards, do the matrices matter \n
        the one at index 0 is there just for padding purposes \n
        important: the input layer is the 0-th layer
        """
        self.sum_dW.append(np.zeros((1, 1)))
        for i in range (1, len(layers)):
            self.sum_dW.append(np.zeros((layers[i], layers[i - 1])))

        self.b:list[np.ndarray] = []
        """
        self.b[i] contains the biases of the i-th layer \n
        important: the input layer is the 0-th layer \n
        the one at index 0 is there just for padding purposes
        """
        for layer_dim in layers:
            self.b.append(np.zeros((layer_dim, 1)))

        self.sum_db:list[np.ndarray] = []
        """
        self.b[i] contains the sum of the partial derivatives column vectors (over all the backpropagations) for the biases of the i-th layer \n
        important: the input layer is the 0-th layer \n
        the one at index 0 is there just for padding purposes
        """
        for layer_dim in layers:
            self.sum_db.append(np.zeros((layer_dim, 1)))

    def save_state(self, file_name="model.json"):
        with open(file_name, "w") as f:
            json.dump({"weights": [w.tolist() for w in self.W], "biases": [b.tolist() for b in self.b]}, f)
        
    def load_state(self, file_name="model.json"):
        with open(file_name, "r") as f:
            data = json.load(f)
            self.W = [np.array(w) for w in data["weights"]]
            self.b = [np.array(b) for b in data["biases"]]

    @staticmethod
    def ReLU(z:np.ndarray):
        return np.maximum(z, 0)

    @staticmethod
    def ReLU_derivative(z:np.ndarray):
        return (z > 0).astype(float)

    @staticmethod
    def Softmax(z:np.ndarray):
        # so we don't have values which are too big
        # we divide both the numerator and denominator by e^{max_val from z} 
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)
    
    @staticmethod
    def Compute_Loss(a:np.ndarray, y:np.ndarray):
        return np.sum(0.5 * ((a - y) ** 2))

    def copy_weights(self, x):
        self.W = [W.copy() for W in x.W]
        self.b = [b.copy() for b in x.b]
        self.sum_db = [sum_db.copy() for sum_db in x.sum_db]
        self.sum_dW = [sum_dW.copy() for sum_dW in x.sum_dW]

    def Predict(self, x:np.ndarray):
        a = x
        for l in range (1, self.nr_layers):
            if l != self.nr_layers - 1:
                a = NeuralNetwork.ReLU(self.W[l] @ a + self.b[l])
            else:
                a = self.W[l] @ a + self.b[l]

        return a

    def Forward_Pass(self, x:np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        a = []
        z = []
        a.append(x)
        z.append(None)
        for l in range (1, self.nr_layers):
            z.append(self.W[l] @ a[l - 1] + self.b[l])
            if l != self.nr_layers - 1:
                a.append(NeuralNetwork.ReLU(z[l]))
            else:
                a.append(z[l])

        return a, z
     
    def Backpropagation(self, a:list[np.ndarray], z:list[np.ndarray], y:np.ndarray):
        dz = a[self.nr_layers - 1] - y
        self.sum_db[self.nr_layers - 1] += dz
        self.sum_dW[self.nr_layers - 1] += dz @ a[self.nr_layers - 2].transpose()
        da = self.W[self.nr_layers - 1].transpose() @ dz

        for l in range (self.nr_layers - 2, 0, -1):
            delta = da * NeuralNetwork.ReLU_derivative(z[l])
            self.sum_db[l] += delta
            self.sum_dW[l] += delta @ a[l - 1].transpose()
            da = self.W[l].transpose() @ delta

    def Update(self, learning_rate, nr_samples):
        if nr_samples == 0:
            return
        for l in range (1, self.nr_layers):
            self.b[l] -= self.sum_db[l] * (learning_rate * (1 / nr_samples))
            self.W[l] -= self.sum_dW[l] * (learning_rate * (1 / nr_samples))
            self.sum_db[l].fill(0)
            self.sum_dW[l].fill(0)  

    def Train_Batch(self, x:list[np.ndarray], y:list[np.ndarray], learning_rate):
        nr_samples = len(x)
        if nr_samples == 0:
            return 0
        
        batch_loss = 0
        for sample in range(nr_samples):
            a, z = self.Forward_Pass(x[sample])
            batch_loss += self.Compute_Loss(a[self.nr_layers - 1], y[sample])
            self.Backpropagation(a, z, y[sample])

        self.Update(learning_rate, nr_samples)

        return batch_loss / nr_samples