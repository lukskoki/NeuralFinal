import numpy as np
import os
class Activation_Softmax:
 # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities


    def to_dict(self):
        return {'type': 'Softmax'}

    @classmethod
    def from_dict(cls, layer_dict):
        return cls()
        

class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
    # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss
    def regularization_loss(self, layer):
        reg_loss = 0

        if layer.weight_reg_l1 > 0:
            reg_loss += layer.weight_reg_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_reg_l2 > 0:
            reg_loss += layer.weight_reg_l2 * np.sum(layer.weights * layer.weights)
        
        if layer.bias_reg_l1 > 0:
            reg_loss += layer.bias_reg_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_reg_l1 > 0:
            reg_loss += layer.bias_reg_l2 * np.sum(layer.biases * layer.biases)
        
        return reg_loss
    
    
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        y_true_clipped = np.clip(y_true, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ] # Gets the probabilities for each correct class


        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true_clipped,
                axis=1
            ) # With this, all values which are not in the true position will get zeroed and only the true value will survive
            # For example, [0.2, 0.3, 0.4, 0.1] and [0,0,1,0] the value which will survive is 0.4

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
    
    def to_dict(self):
        return {'type': 'Loss'}
    
    @classmethod
    def from_dict(cls, layer_dict):
        return cls()

class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons, 
                 weight_reg_l1 = 0, weight_reg_l2 = 0,
                 bias_reg_l1 = 0, bias_reg_l2 = 0, weights = 0, biases = 0):

        if weights == 0:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        else:
            self.weights = weights
        if biases == 0:
            self.biases = np.zeros((1, n_neurons)) 
        else:
            self.biases = biases
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        # Set reg strength

        self.weight_reg_l1 = weight_reg_l1
        self.weight_reg_l2 = weight_reg_l2
        self.bias_reg_l1 = bias_reg_l1
        self.bias_reg_l2 = bias_reg_l2

        # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
    # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

        # Gradients on regularization
        if self.weight_reg_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_reg_l1 * dL1
        
        if self.weight_reg_l2 > 0:
            self.dweights += 2 * self.weight_reg_l2 * self.weights
        
        if self.bias_reg_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_reg_l1 * dL1
        
        if self.bias_reg_l2 > 0:
            self.dbiases += 2 * self.bias_reg_l2 * self.biases
    
    def to_dict(self):
        return {
            'type': 'Dense',
            'input_size': self.n_inputs,
            'output_size': self.n_neurons,
            'weight_reg_l1': self.weight_reg_l1,
            'bias_reg_l1': self.bias_reg_l1,
            'weight_reg_l2': self.weight_reg_l2,
            'bias_reg_l2': self.bias_reg_l2
        }
    def save_weights(self, directory, layer_index):
        """Save weights and biases to an .npz file"""
        filename = os.path.join(directory, f'dense_layer_{layer_index}.npz')
        np.savez(filename, weights=self.weights, biases=self.biases)
        print(f"Weights and biases saved to {filename}.")

    def load_weights(self, directory, layer_index):
        """Load weights and biases from an .npz file"""
        filename = os.path.join(directory, f'dense_layer_{layer_index}.npz')
        data = np.load(filename)
        self.weights = data['weights']
        self.biases = data['biases']
        print(f"Weights and biases loaded from {filename}.")

    @classmethod
    def from_dict(cls, layer_dict):
        return cls(layer_dict['input_size'], layer_dict['output_size'], layer_dict['weight_reg_l1'], layer_dict['weight_reg_l2'], layer_dict['bias_reg_l1'], layer_dict['bias_reg_l2'])



class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let’s make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
    
    def to_dict(self):
        return {'type': 'ReLU'}

    @classmethod
    def from_dict(cls, layer_dict):
        return cls()

class Activation_Softmax_Loss_CategoricalCrossentropy:
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    
        

    # Forward pass
    def forward(self, inputs, y_true, return_loss = True): # During testing if we dont know the y_true values then we cant calculate loss so we dont need to return it
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        if return_loss:
            return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
    
    def to_dict(self):
        return {'type': 'Softmax_Loss'}
    
    @classmethod
    def from_dict(cls, layer_dict):
        return cls()

class Loss_calculate(Loss):

    
    # Forward pass
    def forward(self, inputs, y_true):
        # Calculate and return loss value
        return self.calculate(inputs, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Lecture 24 - gradient descent with momentum
class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay = 0., momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    
    def update(self,layer):
        if self.momentum:
            # If layer does not contain momentum arrays, create them and fill with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - \
                            self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - \
                            self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        else:
            weight_updates = -self.current_learning_rate*layer.dweights
            bias_updates = -self.current_learning_rate*layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update(self):
        self.iterations += 1
    
    def to_dict(self):
        return {
            'type': 'SGD',
            'learning_rate': self.learning_rate,
            'decay': self.decay,
            'momentum': self.momentum,
            
            
        }

    
    @classmethod
    def from_dict(cls, layer_dict):
        return cls(layer_dict['learning_rate'], layer_dict['decay'], layer_dict['momentum'])
    

class Optimizer_Adagrad:

    def __init__(self, learning_rate=1., decay = 0., epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
    
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update(self,layer):
        
          
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        
        
        layer.weights += -self.current_learning_rate*\
            layer.dweights/ \
                (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate*\
            layer.dbiases/ \
                (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update(self):
        self.iterations += 1
    def to_dict(self):
        return {
            'type': 'Adagrad',
            'learning_rate': self.learning_rate,
            'decay': self.decay,
            'epsilon': self.epsilon,
            
            
        }

    
    @classmethod
    def from_dict(cls, layer_dict):
        return cls(layer_dict['learning_rate'], layer_dict['decay'], layer_dict['epsilon'])
    


class Optimizer_Rmsprop:

    def __init__(self, learning_rate=0.001, decay = 0., epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update(self,layer):
        
          
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = layer.weight_cache*self.rho + (1-self.rho)* layer.dweights**2
        layer.bias_cache =layer.bias_cache*self.rho + (1-self.rho) * layer.dbiases**2
        
        
        
        layer.weights += -self.current_learning_rate*\
            layer.dweights/ \
                (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate*\
            layer.dbiases/ \
                (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update(self):
        self.iterations += 1
    
    def to_dict(self):
        return {
            'type': 'RmsProp',
            'learning_rate': self.learning_rate,
            'decay': self.decay,
            'epsilon': self.epsilon,
            'rho': self.rho,
            
        }

    
    @classmethod
    def from_dict(cls, layer_dict):
        return cls(layer_dict['learning_rate'], layer_dict['decay'], layer_dict['epsilon'], layer_dict['rho'])


class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update(self):
        self.iterations += 1
    
    def to_dict(self):
        return {
            'type': 'Adam',
            'learning_rate': self.learning_rate,
            'decay': self.decay,
            'epsilon': self.epsilon,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2
            
        }

    
    @classmethod
    def from_dict(cls, layer_dict):
        return cls(layer_dict['learning_rate'], layer_dict['decay'], layer_dict['epsilon'], layer_dict['beta_1'], layer_dict['beta_2'])



# Lecture 31 - Dropout layer

class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate
    
    def forward(self, inputs):

        self.inputs = inputs
        # Scaled binart mask
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) /self.rate

        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
    
    def to_dict(self):
        return {
            'type': 'Dropout',
            'rate': self.rate
        }

    @classmethod
    def from_dict(cls, layer_dict):
        return cls(1-layer_dict['rate'])
