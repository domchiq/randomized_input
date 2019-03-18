# randomized_input

As there is always a hunger for data, one possible way is to randomize input and create data samples for 
training. Question is however how good will Neural Networks trained on randomized input perform.

## Parts

    - input_randomizer class
    - Neural Network to test training with randomized input

## Functionality

This is a base code for testing randomized input on MNIST where you can randomize input with changing
horizontal and vertical position of the image. You can also randomize value of individual pixels which should
change shoft lines of the image.

<br>Code as is trains Neural Network on randomized data set and evaluates it on real MNIST data.

## Usage
You can either run the code as is or you can change randomized parts, layers in NN or replace NN with other
machine learning algorythm. You can also add your own randomizing methods to the class and see what performance
can you achieve.

### Input
Input for input_randomizer class
```sh
x_train - list of samples (one sample per case) to be randomized
y_train - labels
randomizer_edge=2 - upper limit of arrays possible to replace on the sides of the image
```
### Output
```sh
probability of the correct prediction by Neural Network trained on randomized input
```