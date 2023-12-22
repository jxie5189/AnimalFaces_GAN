# AnimalFaces_GAN

Unable to fully construct an artifical animal face. But still interesting attempting to implement it. 

Dataset: https://www.kaggle.com/datasets/andrewmvd/animal-faces

This notebook assumes that you have the dataset folder in google drive. The notebook connects to google drive and uses only the 'train' folder from the dataset. File path and labels are extracted from the 'train' folder and added to train_data list, and then converted to a dataframe. the train_data(dataframe) is then separated into 3 different dataframe depending on their label [dog, cat, wild]. 

# Image processing
An Image processing method is made, it opens each path, reshapes it to (28,28) with padding, converts it to an array, and finally normalize it to -1 to 1. Unfortunately, this method takes a long time to process each image. Thus, to save RAM, we use a data generator. 

# Data Augmentation
Data Augmentation is done via ImageDataGenerator. There is a preprocessing method to normalized that data. The generator resizes the image to (28,28), 3 color channel of 'rgb', batch size of 256, x_col is the path of the image and y_col is the label. Data generator are made for each of the 3 labels [dog, cat, wild]. 

# Generator
A sequential generator model is made via tf.keras. The generator will intake an input with of 100 dimensions of noise data. The input is then densed to 7*7*256 neurons, no bias, and uses 'he_normal' kernal initializer. Different initializer were experimented to determine the best color scale. 'he_normal' seems to be the one with the most color contrast. 

The layer is then normalized with Momentum to control how much of the current mini-batch is incorporated into the moving average of the batch statistics and Epsilon to improve numerical stability. The activation function for the network is Relu for non-linearity. 

Afterwards the data is reshape into (7,7) and goes thru 2 main Conv2DTranspose layer with decreasing filters of 128 and 68, each with a (4,4) kernel with a stride of (2,2). The layer is then normalized as before, BatchNormliazation with Momentum(0.8) and Epsilon(0.002) and activation of Relu. 

The final output layer has 3 filters with a (4,4) kernel size and stride of (1,1) and uses ‘Tanh’ activation to give the pixel values in the range of [-1,1]. 

The output shape from the generator is (28,28,3), 28 pixel by 28 pixel with 3 color channels. There are a total of 1,963,776 parameters, but only 1,938,304 trainable parameters.

The color schema help to visualized what spectrum random noise data are. The 3 color channel also help to visualized how random noise data captures each color channels. 

# Discriminator
The discriminator is a simple sequential CNN model, input shape of (28,28,3) and applies 64 filters with kernel size of (5,5) and strides of (2,2). The layer is then applied activation function ‘LeakyRelu’ follow by a dropout rate of 0.3. A second Conv2D applies 128 filters with kernel size of (5,5) and stride of (2,2), ‘LeakyRelu’ activation and dropout rate of 0.3. Finally the layer is flatten and densed into 1 neuron to decide whether the input image is real or fake. The discriminator has a total of 216,065 trainable parameters.

# Loss function
The loss function used is binary crossentropy. 

The discriminator's real loss is determined by 1's vs. the input real output and the fake loss is 0's vs. the fake output. The total loss is the sum of the real loss and the fake loss. 

The generator loss is the comparsion of 0s and the input fake loss. 

# optimizer 
Both model uses 'Adam' optimizer with default learning rate. 

# Training 
Training is split into 2 componenet, the train_step and train. In train, each epoch will undergo a train_step. During train_step, a batch of images is inputted into the train_step method. Noise data is created with the same batch size and dimension as the inputted batch. The noise data is then inputted into the generator. Then the discriminator evaluates the real image as real_output and the evaluates the fake image (from the generator) as fake_output. The generator loss is obtained from the fake_output and the discriminator loss is obtained from the real_output. Each gradient is then obtained, and trainable variables anre gradient is saved into the discriminator and generator. 

Train method takes the images, number of epochs, and number of batches. For each epoch, each batch of images will be passed into the train_step, until all the batches haved passed into train_step. An image is generated and saved via generate_and_save_images method. The generate_and_save_images method takes a model, epoch number, and seed data. The model will be the generator to generate images, the epoch number is used to name the image for tracking purposes, the seed data is random noise (number of examples, noise dimension). 16 images (4x4) will be generated. 





