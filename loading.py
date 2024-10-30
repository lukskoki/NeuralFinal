from Network import Network


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


network = Network.load("data") # Network.load("Name of directory where the data is")

# Test
input = network.process_image("image.jpg")
network.test(input, class_names)
input = network.process_image("image1.jpg")
network.test(input, class_names)







