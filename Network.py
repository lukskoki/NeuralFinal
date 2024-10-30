import numpy as np
import cv2
from Classes import *
import json
import os
import matplotlib.pyplot as plt
class Network:
    def __init__(self, layers=[], optimizer = None):
       
        

        self.optimizer = optimizer
       
        self.layers = layers

    def add(self, layer_class):
        self.layers.append(layer_class)
    def forward(self, input, true_values):
        for index in range(0, len(self.layers)):
            
            if index == 0:
                self.layers[index].forward(input)
            else:
                if (index == len(self.layers) - 1):
                    self.layers[index].forward(self.layers[index-1].output, true_values)
                else:
                    self.layers[index].forward(self.layers[index-1].output)
    
    def backward(self, y_train):
        for index in range(len(self.layers) - 1, -1, -1):
            
            if index == len(self.layers) - 1:
                self.layers[index].backward(self.layers[index].output, y_train) # The last layers should be the loss function which takes 2 params
            else:
                self.layers[index].backward(self.layers[index+1].dinputs)
            
    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        """Save the network configuration (layers) to a JSON file."""
        # Convert each layer to a dictionary representation
        
        layers_config = [layer.to_dict() for layer in self.layers]
        optimizer_arr = [self.optimizer.to_dict()]
        
        
        json_filename = os.path.join(directory, 'network_config.json')
        # Write the list of dictionaries to a JSON file
        with open(json_filename, 'w') as f:
            json.dump({'layers': layers_config, 'optimizer': optimizer_arr}, f)
        print(f"Network configuration saved to {json_filename}.")

        # Save weights and biases for Dense layers separately
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Layer_Dense):
                layer.save_weights(directory, f'train_{i}')
    
    
    @classmethod
    def load(cls, directory):
        """Load the network configuration from a JSON file and create a new Network instance."""
        # Read the configuration from the JSON file
        json_filename = os.path.join(directory, 'network_config.json')
        with open(json_filename, 'r') as f:
            config = json.load(f)
        
        # Extract the layers and recreate them
        layers_config = config.get('layers', [])
        
        optimizer_get = config.get('optimizer', [])
        
        optimizer_arr = []
        
        layers = []

        
       # Recreate training layers
        for i, layer_dict in enumerate(layers_config):
            
            layer_type = layer_dict['type']
           
            if layer_type == 'Dense':
                layer = Layer_Dense.from_dict(layer_dict)
                layer.load_weights(directory, f'train_{i}')
                layers.append(layer)
            elif layer_type == 'Dropout':
                layers.append(Layer_Dropout.from_dict(layer_dict))
            elif layer_type == 'ReLU':
                layers.append(Activation_ReLU.from_dict(layer_dict))
            elif layer_type == 'Softmax':
                layers.append(Activation_Softmax.from_dict(layer_dict))
            elif layer_type == "Loss":
                layers.append(Loss_CategoricalCrossentropy.from_dict(layer_dict))
            elif layer_type == "Softmax_Loss":
               
                layers.append(Activation_Softmax_Loss_CategoricalCrossentropy.from_dict(layer_dict))
            
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        # Load Optimizer
        layer_dict = optimizer_get[0]
        layer_type = layer_dict['type']
        
        if layer_type == 'SGD':
            optimizer_arr.append(Optimizer_SGD.from_dict(layer_dict))
        elif layer_type == 'Adagrad':
            optimizer_arr.append(Optimizer_Adagrad.from_dict(layer_dict))
        elif layer_type == 'RmsProp':
            optimizer_arr.append(Optimizer_Rmsprop.from_dict(layer_dict))
        elif layer_type == 'Adam':
            optimizer_arr.append(Optimizer_Adam.from_dict(layer_dict))

        # Create and return a new Network instance with the loaded layers
        return cls(layers, optimizer_arr[0])
    
    def test(self, input, class_names):
        testing = [] # We dont want to include a dropout layer
        for x in self.layers:
            if not isinstance(x, Layer_Dropout):
                testing.append(x)

        for i in range(0, len(self.layers)):
            
            if i == 0:
                self.layers[i].forward(input)
            elif i == len(self.layers) - 1:
                
                self.layers[i].forward(self.layers[i - 1].output,None, False) # Loss function
                
            else:
                self.layers[i].forward(self.layers[i - 1].output)
        
        predictions = self.layers[len(self.layers)-1].output
    
        print("odgovor je: ", class_names[np.argmax(predictions)])
    
    def optimize(self):
        self.optimizer.pre_update()
        for i in range(0, len(self.layers)):
            if isinstance(self.layers[i], Layer_Dense):
                self.optimizer.update(self.layers[i])

        self.optimizer.post_update()
    
    def train(self,X_train_full, y_train_full, iterations):
        for i in range(iterations):
            self.forward(X_train_full, y_train_full)
            loss_activation = self.layers[len(self.layers)-1]
            loss = loss_activation.loss.calculate(self.layers[len(self.layers)-1].output, y_train_full)

            reg_loss = (loss_activation.loss.regularization_loss(self.layers[0]) + \
                        loss_activation.loss.regularization_loss(self.layers[3]))
            
            total_loss = loss + reg_loss


            predictions = np.argmax(self.layers[len(self.layers)-1].output, axis = 1)
            if len(y_train_full.shape) == 2:
                y_train_full = np.argmax(y_train_full, axis = 1) # If it is one hot encoded return an array where the elements are the correct classes
            
            
            acc = np.mean(predictions == y_train_full)

            if not i % 50:
                print("Loss:", loss, " Acc:", acc," Learning rate: ", self.optimizer.current_learning_rate, " Epoch: ", i)
            
            
            # Backward pass
            self.backward(y_train_full)

            
            # Update weights and biases
            self.optimize()
    def process_image(self, image):
        image = cv2.imread(image)
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to 28x28 pixels
        resized_image = cv2.resize(gray_image, (28, 28))

        resized_image = 255 - resized_image
        plt.imshow(resized_image, cmap='binary')
        plt.axis('off')
        plt.show()
        # Normalize the image (convert pixel values to the range [0, 1])
        normalized_image = resized_image.astype('float32') / 255.0
        # Flatten the image to a vector of size 784
        flattened_image = normalized_image.flatten()
        # If the model expects a batch, reshape it accordingly
        input_image = flattened_image.reshape(1, 784)
        # Now you can pass input_image to your model's predict method
        # With regularization
        return input_image

    def __str__(self):
        """Provide a string representation of the network configuration."""
        layers_descriptions = [layer.to_dict() for layer in self.layers]
        optimizer_arr = [self.optimizer.to_dict()]
        return (f"Network with training layers: {layers_descriptions}\n"
                f"Optimizer: {optimizer_arr}")
