# Deep L-Layer Nueral Network For N-Epochs

from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import pickle

class DNN:
   
    """DNN V.1.0 helps users to create a Deep L-layer Nueral Network without the need to code"""
    
    # Class Level Annotations
    layers_dim: List[int] # Layer dimension will be list of integers containing each layer's dimensions
    num_layers: int
    parameters: Dict[str, np.ndarray]
    grads: Dict[str, np.ndarray]
    caches: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    costs: List[float]


    def __init__(self, layers_dim: List[int]) -> None:
        """           
            Args:
            num_layers: Total number of Layers including input, hidden and output
            layers_dim: A list containing the number of hidden units for each hidden layer, 
                          input and output layer  i.e [input_units, hidden1_units, hidden2_units, output_units]
        """
        self.layers_dim = layers_dim
        if len(self.layers_dim) < 2:
            raise ValueError("Layers should include Input, Hidden and Output Layer i.e. must be greater than 2")
        self.num_layers = len(layers_dim)
        self.parameters = {}
        self.grads = {}
        self.caches = []
        self.costs = []
    
    
    def parameters_initialization(self, seed: int) -> None:
        """ Initializes the Nueral Networks parameters like weights and biases for each layer
            parameters: python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl: weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl: bias vector of shape (layer_dims[l], 1) 

            Args:
            seed: random seed for reproducibility
            num_layers: Total number of Layers including input, hidden and output
            layers_dim: python list containing the number of hidden units for each hidden layer, 
                        input and output layer  i.e [input_units, hidden1_units, hidden2_units, output_units] 
        """
        np.random.seed(seed)
        
        for l in range(1, self.num_layers):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_dim[l], self.layers_dim[l-1]) * 0.01
            self.parameters["b" + str(l)] = np.zeros((self.layers_dim[l], 1))
    
    
    def ff_linear(self, A:np.ndarray, W:np.ndarray, b:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates Z i.e. linear combination of weights and Activations from previous layers
        Args:
        A: activations from previous layer (n_prev, m)
        W: weights (n_curr, n_prev)
        b: biases (n_curr, 1)

        returns: 
        cache: tuple (A, W, b, Z)
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b, Z)
        return cache
    
    
    def ff_activation(self, A_prev: np.ndarray, W:np.ndarray, b:np.ndarray, activation:str) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """ 
        Single Layer's linear forward and activation calculation

        returns: A (post‐activation), linear_cache (A_prev, W, b, Z) 
        
        """
        linear_cache = self.ff_linear(A_prev, W, b)

        if activation == "linear": 
            A = linear_cache[3]
             
        elif activation == "relu":
            A = np.maximum(0, linear_cache[3])
             
        elif activation == "sigmoid":
            A = 1 / (1 + np.exp(-linear_cache[3]))

        elif activation == "tanh":
             A = np.tanh(linear_cache[3])

        else:
            raise ValueError("Invalid Activation Function")

        return A, linear_cache
    
    
    def forward_propagation(self, X: np.ndarray, hidden_activation: str = "relu", final_activation:str = "sigmoid") -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]: 
        """One complete Feedforward loop i.e. linear forward + linear activation over the entire layers 
        
        returns: AL or y_hat (output), caches for backprop
        
        """
        self.caches = []
        A_prev = X
        
        # Feedforward Propagation for hidden layers
        for l in range(1, self.num_layers - 1):
            A_prev, linear_cache = self.ff_activation(A_prev, self.parameters["W" + str(l)], self.parameters["b" + str(l)], activation = hidden_activation )
            self.caches.append(linear_cache)
        
        # Feedforward Propagation for output layer
        AL, final_linear_cache = self.ff_activation(A_prev, self.parameters["W" + str(self.num_layers - 1)], self.parameters["b" + str(self.num_layers - 1)], activation = final_activation )
        self.caches.append(final_linear_cache)
        
        return AL, self.caches
    
    
    def cost(self, AL: np.ndarray, Y: np.ndarray, task: str = "classification") -> float:
        """ Calculates the cost for one feedforward loop
        
        Args:
        AL: probability vector corresponding to your label predictions, shape (1, number of examples)
        Y: true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost: cross-entropy cost
        """
        m = Y.shape[1]
        
        # Compute loss from aL and y.
        epsilon = 1e-8
        if task == "classification":
            cost = - np.sum(Y * np.log(AL + epsilon) + (1-Y) * np.log(1 - AL + epsilon)) / m
        else:
            cost = 1/m * np.sum((AL - Y)**2)
        return np.squeeze(cost)
    

    def bw_linear(self, dZ: np.ndarray, A_prev: np.ndarray, W:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the backward linear for one layer
                
        returns: 
        dA_prev: Activation gradient w.r.t loss function of L-1 layer
        dW: Weight gradient w.r.t loss function of current L layer
        db: Bias gradient w.r.t loss function of current L layer
        """
        m = A_prev.shape[1]
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis = 1, keepdims= True)
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db
    
    
    def bw_activation(self, dA: np.ndarray, cache:Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] , activation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the backward linear and activation for one layer

        Args:
        cache: linear cache of respective layer (A_prev,W,b,Z) 
        dZ: Z gradient w.r.t loss function of current L layer (Used to calculate dA_prev, dW, db)
        
        returns:
        dA_prev: Activation gradient w.r.t loss function of L-1 layer
        dW: Weight gradient w.r.t loss function of current L layer
        db: Bias gradient w.r.t loss function of current L layer
        """
        A_prev, W, b, Z = cache
        
        if activation == "relu":
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0

        elif activation == "sigmoid":
            s = 1 / (1 + np.exp(-Z))
            dZ = dA * s * (1 - s)

        elif activation == "tanh":
            t = np.tanh(Z)
            dZ = dA * (1 - t**2)

        elif activation == "linear":
            dZ = dA

        dA_prev, dW, db = self.bw_linear(dZ, A_prev, W)

        return dA_prev, dW, db
    

    def backward_propagation(self, AL: np.ndarray, Y:np.ndarray, hidden_activation: str = "relu", task: str = "classification") ->Dict[str, np.ndarray]:
        """
        Complete One Backward Propagation loop i.e. linear backward + activation over the entire layers
        returns: grads dict
        """
        m = Y.shape[1]
        current_cache = self.caches[-1]

        if task == "classification":
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
            dA_prev_temp, dW_temp, db_temp = self.bw_activation(dAL, current_cache, activation = "sigmoid")
        else:
            dAL = 2/m * (AL - Y)
            dA_prev_temp, dW_temp, db_temp = self.bw_activation(dAL, current_cache, activation = "linear")

        self.grads["dA" + str(self.num_layers-2)] = dA_prev_temp
        self.grads["dW" + str(self.num_layers-1)] = dW_temp
        self.grads["db" + str(self.num_layers-1)] = db_temp


        for l in reversed(range(1, self.num_layers - 1)):
            current_cache = self.caches[l-1]
            dA_prev_temp, dW_temp, db_temp = self.bw_activation(dA_prev_temp, current_cache, hidden_activation)
            self.grads["dA" + str(l-1)] = dA_prev_temp 
            self.grads["dW" + str(l)] = dW_temp 
            self.grads["db" + str(l)] = db_temp

        return self.grads
    
    def update_parameters(self, learning_rate: float) -> Dict[str, np.ndarray]:
        """ Updating the parameters i.e. weights and biases after calculating the gradients
        
        returns: updated parameters in dict
        """
        for l in range(1, self.num_layers):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * self.grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * self.grads["db" + str(l)]

        return self.parameters
    
    
    def train(self, X: np.ndarray, Y:np.ndarray, task:str, epochs: int, learning_rate: float, hidden_activation:str = "relu", final_activation: str = "sigmoid", seed:int = 42, print_cost: bool = False) ->Tuple[Dict[str, np.ndarray], List[float]] :
        """
        Training a Nueral Network i.e. Feedforward Propogation + Cost Calculation + Backward Propogation + Updating Parameters for n epochs

        returns: learned parameters and cost history
        """
        self.task = task
        self.hidden_activation = hidden_activation
        self.final_activation = final_activation
        self.parameters_initialization(seed)

        for i in range(epochs):
            AL, self.caches = self.forward_propagation(X, hidden_activation, final_activation)
            current_epoch_cost = self.cost(AL, Y, task)
            self.costs.append(current_epoch_cost)
            self.backward_propagation(AL, Y, hidden_activation, task)
            self.update_parameters(learning_rate)
            if print_cost == True and i % 100 == 0:
                print(f"The Cost of {i} Epoch is {current_epoch_cost}")
        return self.parameters, self.costs
    
    
    def plot_costs(self, learning_rate: float) -> None:
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per epochs)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the sigmoid output AL for each example: shape (1, m),
        i.e. P(y=1 | x).
        """
        AL, _ = self.forward_propagation(X, self.hidden_activation, self.final_activation)
        return AL            


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: predictions for unseen data
        
        """
        AL = self.predict_proba(X)
        
        if self.task == "classification":
            return (AL > 0.5).astype(int)
        else:
            return AL    
    
    def save_parameters(self, path: str) -> None:
        """
        Save the architecture (layers_dim) and learned parameters (weights & biases) to disk.

        Args:
            path: filesystem path where parameters will be written, e.g. 'model.pkl'
        """
        state = {
            "layers_dim": self.layers_dim,
            "parameters": self.parameters
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model architecture and parameters saved to \n '{path}'")

    def load_parameters(self, path: str) -> None:
        """
        Load architecture and parameters from disk into this model.

        This will overwrite this instance's layers_dim, num_layers, and parameters.

        Args:
            path: path to the pickle file, e.g. 'dnn_checkpoint.pkl'
        """

        with open(path, "rb") as f:
            state = pickle.load(f)

        # Restoring the Architecture
        self.layers_dim = state["layers_dim"]
        self.num_layers = len(self.layers_dim)

        # Restoring the learned parameters
        self.parameters = state["parameters"]

        # Reset any training‐specific buffers
        self.grads = {}
        self.caches = []
        self.costs = []

        print(f"Model restored from '{path}'")

    

   
