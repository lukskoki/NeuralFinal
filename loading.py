import numpy as np
import cv2
from Classes import Layer_Dense, Layer_Dropout, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Activation_Softmax, Loss_CategoricalCrossentropy
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
import os
from Network import Network


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


network = Network.load("prvitest")

# Test
input = network.process_image("image.jpg")
network.test(input, class_names)
input = network.process_image("image1.jpg")
network.test(input, class_names)

#network.save('data')





