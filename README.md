# Handwritten-Digit-Recognition
Building a simple Convolutional Neural Network using mnist data set to recognize handwritten digits.

## Dataset:

MNIST (“Modified National Institute of Standards and Technology”) is the de facto “Hello World” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

## Data Processing:

The data set contains 60,000 traning images and 10000 testing images. Here I split the data into training and testing datasets respectively. The train_X & test_X contains grayscale codes while test_y & train_y contains labels from 0–9 which represents the numbers.

When you check the shape of the dataset to see if it is compatible to use in for CNN. You can see we will (60000,28,28) as our result which means that we have 60000 images in our dataset and size of each image is 28 * 28 pixel.

To use Keras API we need a 4-dimensional array but we can see from above that we have a 3-dimension numpy array.

## Building the Model:

I use the Keras API to build the model hence I have a Tensorflow background.I import the Sequential Model from Keras and add Conv2D, MaxPooling, Flatten, Dropout, and Dense layers.

Dropout layers fight with the overfitting by disregarding some of the neurons while training while Flatten layers flatten 2D arrays to 1D array before building the fully connected layers.

Compiling and fitting the Model:

So far, we have created an non-optimized empty CNN. Then I set an optimizer with a given loss function which uses a metric and fit the model by using our train data. The ADAM optimizer is said to outperform the other optimizers, that’s why I used that.

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)
```

Here we get pretty high accuracy with just 10 epochs. Since the dataset doesn’t need heavy computational power you can play around with the number of epochs you can also play around with the optimizer, loss function and metrics.

## Model Evaluation:

When this model is evaluated we see that just 10 epochs gave use the accuracy of 98.39% at a very low loss.
